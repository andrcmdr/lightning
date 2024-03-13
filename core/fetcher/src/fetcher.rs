use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use affair::{Socket, Task};
use anyhow::{anyhow, Context, Result};
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::{
    Blake3Hash,
    FetcherRequest,
    FetcherResponse,
    ImmutablePointer,
    ServerRequest,
};
use lightning_interfaces::{
    BlockstoreInterface,
    BlockstoreServerInterface,
    BlockstoreServerSocket,
    ConfigConsumer,
    FetcherInterface,
    FetcherSocket,
    OriginProviderInterface,
    OriginProviderSocket,
    ResolverInterface,
    WithStartAndShutdown,
};
use lightning_metrics::increment_counter;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::config::Config;
use crate::origin::{OriginFetcher, OriginRequest};

pub(crate) type Uri = Vec<u8>;

pub struct Fetcher<C: Collection> {
    inner: Arc<FetcherInner<C>>,
    is_running: Arc<AtomicBool>,
    socket: FetcherSocket,
    shutdown_tx: mpsc::Sender<()>,
}

impl<C: Collection> FetcherInterface<C> for Fetcher<C> {
    /// Initialize the fetcher.
    fn init(
        config: Self::Config,
        blockstore: C::BlockstoreInterface,
        blockstore_server: &C::BlockstoreServerInterface,
        resolver: C::ResolverInterface,
        origin: &C::OriginProviderInterface,
    ) -> anyhow::Result<Self> {
        let (socket, socket_rx) = Socket::raw_bounded(2048);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(10);
        let inner = FetcherInner::<C> {
            config,
            socket_rx: Arc::new(Mutex::new(Some(socket_rx))),
            origin_socket: origin.get_socket(),
            blockstore,
            blockstore_server_socket: blockstore_server.get_socket(),
            resolver,
            shutdown_rx: Arc::new(Mutex::new(Some(shutdown_rx))),
        };

        Ok(Self {
            inner: Arc::new(inner),
            is_running: Arc::new(AtomicBool::new(false)),
            socket,
            shutdown_tx,
        })
    }

    fn get_socket(&self) -> FetcherSocket {
        self.socket.clone()
    }
}

impl<C: Collection> WithStartAndShutdown for Fetcher<C> {
    fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    async fn start(&self) {
        if !self.is_running() {
            let inner = self.inner.clone();
            let is_running = self.is_running.clone();
            tokio::spawn(async move {
                inner.start().await;
                is_running.store(false, Ordering::Relaxed);
            });
            self.is_running.store(true, Ordering::Relaxed);
        } else {
            error!("Cannot start fetcher because it is already running");
        }
    }

    async fn shutdown(&self) {
        self.shutdown_tx
            .send(())
            .await
            .expect("Failed to send shutdown signal");
    }
}

struct FetcherInner<C: Collection> {
    config: Config,
    #[allow(clippy::type_complexity)]
    socket_rx: Arc<Mutex<Option<mpsc::Receiver<Task<FetcherRequest, FetcherResponse>>>>>,
    origin_socket: OriginProviderSocket,
    blockstore: C::BlockstoreInterface,
    blockstore_server_socket: BlockstoreServerSocket,
    resolver: C::ResolverInterface,
    shutdown_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
}

impl<C: Collection> FetcherInner<C> {
    async fn start(self: Arc<Self>) {
        let mut socket_rx = self.socket_rx.lock().unwrap().take().unwrap();

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (tx, rx) = mpsc::channel(128);
        let origin_fetcher = OriginFetcher::<C>::new(
            self.config.max_conc_origin_req,
            self.origin_socket.clone(),
            rx,
            self.resolver.clone(),
            shutdown_rx,
        );
        tokio::spawn(async move {
            origin_fetcher.start().await;
        });
        let mut shutdown_rx = self.shutdown_rx.lock().unwrap().take().unwrap();

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    shutdown_tx.send(()).expect("Failed to send shutdown signal to origin fetcher");
                    break;
                }
                task = socket_rx.recv() => {
                    let Some(task) = task else {
                        continue;
                    };
                    match task.request.clone() {
                        FetcherRequest::Put { pointer } => {
                            let tx = tx.clone();
                            let inner = self.clone();
                            tokio::spawn(async move {
                                let res = inner.put(pointer, &tx).await;
                                if res.is_err() {
                                    increment_counter!(
                                        "fetcher_put_request_failed",
                                        Some("Counter for failed put requests for origin content")
                                    );
                                } else {
                                    increment_counter!(
                                        "fetcher_put_request_succeed",
                                        Some("Counter for successful put requests for origin content")
                                    );
                                }
                                task.respond(FetcherResponse::Put(res));
                            });
                        }
                        FetcherRequest::Fetch { hash } => {
                            let tx = tx.clone();
                            let inner = self.clone();
                            tokio::spawn(async move {
                                let res = inner.fetch(hash, &tx).await;
                                if res.is_err() {
                                    increment_counter!(
                                        "fetcher_fetch_request_failed",
                                        Some("Counter for failed fetch requests for native content")
                                    );
                                } else {
                                    increment_counter!(
                                        "fetcher_fetch_request_succeed",
                                        Some("Counter for successful fetch requests for native content")
                                    );
                                }
                                task.respond(FetcherResponse::Fetch(res));
                            });
                        }
                    }
                }
            }
        }
        *self.socket_rx.lock().unwrap() = Some(socket_rx);
        *self.shutdown_rx.lock().unwrap() = Some(shutdown_rx);
    }

    /// Fetches the data from the corresponding origin, puts it in the blockstore,
    /// and stores the mapping using the resolver. If pulling a origin fails, it will not ,
    /// the data will not be fetched from origin again.
    #[inline(always)]
    async fn put(
        &self,
        pointer: ImmutablePointer,
        origin_tx: &mpsc::Sender<OriginRequest>,
    ) -> anyhow::Result<[u8; 32]> {
        if let Some(hash) = self.resolver.get_blake3_hash(pointer.clone()).await {
            // If we know about a mapping, forward the call to fetch which
            // will attempt to pull from multiple sources.
            return self.fetch(hash, origin_tx).await.map(|_| hash);
        }

        // Otherwise, try to fetch directly from the origin
        self.fetch_origin(pointer, origin_tx).await
    }

    #[inline(always)]
    async fn fetch_origin(
        &self,
        pointer: ImmutablePointer,
        origin_tx: &mpsc::Sender<OriginRequest>,
    ) -> Result<[u8; 32]> {
        let (response_tx, response_rx) = oneshot::channel();
        origin_tx
            .send(OriginRequest {
                pointer,
                response: response_tx,
            })
            .await
            .expect("Failed to send origin request");

        let res = response_rx
            .await?
            .recv()
            .await?
            .context("failed to fetch from origin");

        if res.is_ok() {
            increment_counter!(
                "fetcher_from_origin",
                Some("Counter for content that was fetched from an origin")
            );
        } else {
            increment_counter!(
                "fetcher_from_origin_failed",
                Some("Counter for a failed attempts to fetch from an origin")
            );
        }

        res
    }

    /// Attempt to fetch the blake3 content. First, we check the blockstore,
    /// then iterate through the provider records, requesting from the provider,
    /// then falling back to the record's immutable pointer.
    #[inline(always)]
    async fn fetch(&self, hash: Blake3Hash, origin_tx: &mpsc::Sender<OriginRequest>) -> Result<()> {
        if self.blockstore.get_tree(&hash).await.is_some() {
            increment_counter!(
                "fetcher_from_cache",
                Some("Counter for content that was already cached locally")
            );
            return Ok(());
        } else if let Some(pointers) = self.resolver.get_origins(hash) {
            for res_pointer in pointers {
                debug_assert_eq!(res_pointer.hash, hash);
                if self.blockstore.get_tree(&hash).await.is_some() {
                    // in case we have the file
                    increment_counter!(
                        "fetcher_from_cache",
                        Some("Counter for content that was already cached locally")
                    );
                    return Ok(());
                }

                // Try to get the content from the peer that advertised the record.
                // TODO: Maybe use indexer for getting peers instead
                let mut res = self
                    .blockstore_server_socket
                    .run(ServerRequest {
                        hash,
                        peer: res_pointer.originator,
                    })
                    .await
                    .expect("Failed to send request to blockstore server");
                let res = res
                    .recv()
                    .await
                    .expect("Failed to receive response from blockstore server");

                if res.is_ok() {
                    increment_counter!(
                        "fetcher_from_peer",
                        Some("Counter for content that was fetched from a peer")
                    );
                    return Ok(());
                } else {
                    increment_counter!(
                        "fetcher_from_peer_failed",
                        Some("Counter for failed attempts to fetch from a peer")
                    );
                }

                // If not, attempt to pull from the origin. This strikes a balance between trying
                // to fetch from a bunch of peers vs going to the origin right away.
                if self
                    .fetch_origin(res_pointer.pointer, origin_tx)
                    .await
                    .is_ok()
                {
                    return Ok(());
                }
            }
        }
        Err(anyhow!("Failed to resolve hash"))
    }
}

impl<C: Collection> ConfigConsumer for Fetcher<C> {
    const KEY: &'static str = "fetcher";

    type Config = Config;
}
