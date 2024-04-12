mod config;
mod handler;

use async_channel::{Receiver, Sender};
use async_trait::async_trait;
use axum::routing::any;
use axum::Router;
use bytes::Bytes;
pub use config::Config;
use fn_sdk::header::TransportDetail;
use lightning_interfaces::ExecutorProviderInterface;
use lightning_schema::handshake::{
    HandshakeRequestFrame,
    HandshakeResponse,
    RequestFrame,
    ResponseFrame,
    TerminationReason,
};
use tokio::sync::oneshot;
use tracing::warn;

use crate::shutdown::ShutdownWaiter;
use crate::transports::{Transport, TransportReceiver, TransportSender};

pub struct HttpTransport {}

#[async_trait]
impl Transport for HttpTransport {
    type Config = Config;
    type Sender = HttpSender;
    type Receiver = HttpReceiver;

    async fn bind<P: ExecutorProviderInterface>(
        _: ShutdownWaiter,
        _: Self::Config,
    ) -> anyhow::Result<(Self, Option<Router>)> {
        let router = Router::new().route("/services/:service/*path", any(handler::handler::<P>));
        Ok((Self {}, Some(router)))
    }

    async fn accept(&mut self) -> Option<(HandshakeRequestFrame, Self::Sender, Self::Receiver)> {
        unreachable!()
    }
}

pub enum Service {
    Fetcher,
    Js,
}

pub struct HttpSender {
    service: Service,
    frame_tx: Sender<Option<RequestFrame>>,
    body_tx: Sender<anyhow::Result<Bytes>>,
    current_write: usize,
    termination_tx: Option<oneshot::Sender<TerminationReason>>,
}

impl HttpSender {
    pub fn new(
        service: Service,
        frame_tx: Sender<Option<RequestFrame>>,
        body_tx: Sender<anyhow::Result<Bytes>>,
        termination_tx: oneshot::Sender<TerminationReason>,
    ) -> Self {
        Self {
            service,
            frame_tx,
            body_tx,
            current_write: 0,
            termination_tx: Some(termination_tx),
        }
    }

    #[inline(always)]
    fn inner_send(&mut self, bytes: Bytes) {
        if let Err(e) = self.body_tx.try_send(Ok(bytes)) {
            warn!("payload dropped, failed to send to write loop: {e}");
        }
    }

    #[inline(always)]
    fn close(&mut self) {
        if self.frame_tx.try_send(None).is_err() {
            warn!("failed to send close signal to receiver");
        }
    }
}

impl TransportSender for HttpSender {
    fn send_handshake_response(&mut self, _: HandshakeResponse) {
        unimplemented!()
    }

    fn send(&mut self, _: ResponseFrame) {
        unimplemented!()
    }

    fn terminate(mut self, reason: TerminationReason) {
        if let Some(reason_sender) = self.termination_tx.take() {
            let _ = reason_sender.send(reason);
        }
    }

    fn start_write(&mut self, len: usize) {
        self.termination_tx.take();

        debug_assert!(
            self.current_write == 0,
            "data should be written completely before calling start_write again"
        );

        self.current_write = len;
    }

    fn write(&mut self, buf: Bytes) -> anyhow::Result<usize> {
        let len = buf.len();

        debug_assert!(self.current_write != 0);
        debug_assert!(self.current_write >= len);

        self.current_write -= len;
        self.inner_send(buf);

        Ok(len)
    }
}

pub struct HttpReceiver {
    inner: Receiver<Option<RequestFrame>>,
    detail: Option<TransportDetail>,
}

impl HttpReceiver {
    pub fn new(inner: Receiver<Option<RequestFrame>>, detail: TransportDetail) -> Self {
        Self {
            inner,
            detail: Some(detail),
        }
    }
}

#[async_trait]
impl TransportReceiver for HttpReceiver {
    fn detail(&mut self) -> TransportDetail {
        self.detail
            .take()
            .expect("HTTP Transport detail already taken.")
    }

    async fn recv(&mut self) -> Option<RequestFrame> {
        self.inner.recv().await.ok().flatten()
    }
}
