mod signal;
mod worker;

use std::net::SocketAddr;
use std::sync::Arc;

use affair::{Executor, TokioSpawn};
use async_trait::async_trait;
use log::error;
use serde::{Deserialize, Serialize};
use webrtc::data_channel::data_channel_message::DataChannelMessage;
use webrtc::data_channel::RTCDataChannel;

use self::signal::start_signaling_server;
use self::worker::IncomingConnectionWorker;
use super::{Transport, TransportReceiver, TransportSender};
use crate::schema::{self, HandshakeRequestFrame, RequestFrame};
use crate::shutdown::ShutdownWaiter;

#[derive(Serialize, Deserialize, Clone)]
pub struct WebRtcConfig {
    /// Address to listen on for the signaling server. This is used to receive and respond to
    /// incoming RTC Session Descriptions, to negotiate a new SRTP connection.
    pub signal_address: SocketAddr,
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            signal_address: ([0, 0, 0, 0], 4210).into(),
        }
    }
}

/// A WebRTC Transport. Spawns a HTTP signaling server, and binds to ephemeral UDP ports per
/// peer connection.
pub struct WebRtcTransport {
    /// Receiver for incoming DataChannels from peer connections.
    conn_rx: tokio::sync::mpsc::Receiver<(HandshakeRequestFrame, Arc<RTCDataChannel>)>,
}

#[async_trait]
impl Transport for WebRtcTransport {
    type Config = WebRtcConfig;
    type Sender = WebRtcSender;
    type Receiver = WebRtcReceiver;

    async fn bind(waiter: ShutdownWaiter, config: Self::Config) -> anyhow::Result<Self> {
        log::info!("Binding WebRTC transport on {}", config.signal_address);

        let (conn_tx, conn_rx) = tokio::sync::mpsc::channel(16);

        // Spawn a worker for handling new connection setup.
        let worker = IncomingConnectionWorker { conn_tx };
        let socket = TokioSpawn::spawn_async(worker);

        // Spawn a HTTP server for accepting incoming SDP requests.
        tokio::spawn(async move {
            start_signaling_server(waiter, config, socket)
                .await
                .expect("Failed to setup server");
        });

        Ok(Self { conn_rx })
    }

    async fn accept(
        &mut self,
    ) -> Option<(schema::HandshakeRequestFrame, Self::Sender, Self::Receiver)> {
        let (req, data_channel) = self.conn_rx.recv().await?;

        // Setup message receiver channel and callback method.
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        data_channel.on_message(Box::new(
            move |msg: DataChannelMessage| match RequestFrame::decode(&msg.data) {
                Ok(frame) => {
                    let tx = tx.clone();
                    Box::pin(async move {
                        if let Err(e) = tx.send(frame).await {
                            error!("failed to send message to receiver: {e}");
                        }
                    })
                },
                Err(e) => {
                    error!("failed to decode message: {e}");
                    Box::pin(async {})
                },
            },
        ));

        let receiver = WebRtcReceiver(rx);
        let sender = WebRtcSender(data_channel);

        Some((req, sender, receiver))
    }
}

/// Sender for a webrtc connection.
pub struct WebRtcSender(Arc<RTCDataChannel>);

macro_rules! webrtc_send {
    ($t1:expr, $t2:expr) => {
        let data_channel = $t1.0.clone();
        let bytes = $t2.encode();
        tokio::spawn(async move {
            if let Err(e) = data_channel.send(&bytes).await {
                error!("failed to send message to peer: {e}");
            };
        });
    };
}

impl TransportSender for WebRtcSender {
    fn send_handshake_response(&mut self, frame: schema::HandshakeResponse) {
        webrtc_send!(self, frame);
    }

    fn send(&mut self, frame: schema::ResponseFrame) {
        webrtc_send!(self, frame);
    }
}

/// Receiver for a webrtc connection.
pub struct WebRtcReceiver(tokio::sync::mpsc::Receiver<schema::RequestFrame>);

#[async_trait]
impl TransportReceiver for WebRtcReceiver {
    async fn recv(&mut self) -> Option<schema::RequestFrame> {
        self.0.recv().await
    }
}
