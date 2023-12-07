use std::marker::PhantomData;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use affair::{Socket, Task};
use anyhow::Result;
use async_trait::async_trait;
use fleek_crypto::{NodePublicKey, NodeSecretKey, SecretKey};
use lightning_interfaces::dht::{DhtInterface, DhtSocket};
use lightning_interfaces::infu_collection::{c, Collection};
use lightning_interfaces::types::{DhtRequest, DhtResponse, KeyPrefix, TableEntry};
use lightning_interfaces::{
    ConfigConsumer,
    ReputationAggregatorInterface,
    SignerInterface,
    WithStartAndShutdown,
};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, Notify};

use crate::config::Config;
use crate::network::network;
use crate::node::NodeInfo;
use crate::task::bootstrap::Bootstrapper;
use crate::{api, store, table, task};

/// Builds the DHT.
pub struct Builder<C: Collection> {
    config: Config,
    nodes: Vec<NodeInfo>,
    network_secret_key: NodeSecretKey,
    buffer_size: Option<usize>,
    rep_reporter: c!(C::ReputationAggregatorInterface::ReputationReporter),
    local_rep_query: c![C::ReputationAggregatorInterface::ReputationQuery],
    _marker: PhantomData<C>,
}

impl<C: Collection> Builder<C> {
    /// Returns a new [`Builder`].
    pub fn new(
        network_secret_key: NodeSecretKey,
        config: Config,
        rep_reporter: c!(C::ReputationAggregatorInterface::ReputationReporter),
        local_rep_query: c![C::ReputationAggregatorInterface::ReputationQuery],
    ) -> Self {
        let nodes: Vec<NodeInfo> = config
            .bootstrappers
            .iter()
            .map(|b| NodeInfo {
                key: b.network_public_key,
                address: b.address,
                last_responded: None,
            })
            .collect();
        Self {
            config,
            nodes,
            network_secret_key,
            buffer_size: None,
            rep_reporter,
            local_rep_query,
            _marker: PhantomData,
        }
    }

    /// Add node which will be added to routing table.
    pub fn add_node(&mut self, key: NodePublicKey, address: SocketAddr) {
        self.nodes.push(NodeInfo {
            key,
            address,
            last_responded: None,
        });
    }

    /// Set buffer size for tasks.
    pub fn set_buffer_size(&mut self, size: usize) {
        self.buffer_size = Some(size);
    }

    /// Build and initiates the DHT.
    pub fn build(self) -> Result<Dht<C>> {
        let buffer_size = self.buffer_size.unwrap_or(10_000);

        let (socket, rx) = Socket::raw_bounded(2048);

        let (bootstrap_socket, bootstrap_rx) = Socket::raw_bounded(2048);

        Ok(Dht {
            socket,
            socket_rx: Arc::new(Mutex::new(Some(rx))),
            nodes: Arc::new(Mutex::new(Some(self.nodes))),
            buffer_size,
            address: self.config.address,
            network_secret_key: self.network_secret_key,
            bootstrap_socket,
            bootstrap_rx: Arc::new(Mutex::new(Some(bootstrap_rx))),
            is_running: Arc::new(Mutex::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
            rep_reporter: self.rep_reporter,
            local_rep_query: self.local_rep_query,
            collection: PhantomData,
        })
    }
}

/// Maintains the DHT.
#[allow(clippy::type_complexity)]
pub struct Dht<C: Collection> {
    socket: DhtSocket,
    socket_rx: Arc<Mutex<Option<mpsc::Receiver<Task<DhtRequest, DhtResponse>>>>>,
    buffer_size: usize,
    address: SocketAddr,
    network_secret_key: NodeSecretKey,
    bootstrap_socket: Socket<(), Result<()>>,
    bootstrap_rx: Arc<Mutex<Option<mpsc::Receiver<Task<(), Result<()>>>>>>,
    nodes: Arc<Mutex<Option<Vec<NodeInfo>>>>,
    is_running: Arc<Mutex<bool>>,
    shutdown_notify: Arc<Notify>,
    rep_reporter: c!(C::ReputationAggregatorInterface::ReputationReporter),
    local_rep_query: c![C::ReputationAggregatorInterface::ReputationQuery],
    collection: PhantomData<C>,
}

impl<C: Collection> Dht<C> {
    /// Return one value associated with the given key.
    pub async fn get(&self, prefix: KeyPrefix, key: &[u8]) -> Option<TableEntry> {
        match self
            .socket
            .run(DhtRequest::Get {
                prefix,
                key: key.to_vec(),
            })
            .await
        {
            Ok(DhtResponse::Get(value)) => value,
            Err(e) => {
                tracing::error!("failed to get entry for key {key:?}: {e:?}");
                None
            },
            Ok(_) => unreachable!(),
        }
    }

    /// Put a key-value pair into the DHT.
    pub fn put(&self, prefix: KeyPrefix, key: &[u8], value: &[u8]) {
        let socket = self.socket.clone();
        let key = key.to_vec();
        let value = value.to_vec();
        tokio::spawn(async move {
            if let Err(e) = socket.enqueue(DhtRequest::Put { prefix, key, value }).await {
                tracing::error!("failed to put entry: {e:?}");
            }
        });
    }

    /// Start bootstrap task.
    /// If bootstrapping is in process, this request will be ignored.
    pub async fn bootstrap(&self) -> Result<()> {
        self.bootstrap_socket
            .run(())
            .await
            .map_err(|e| anyhow::anyhow!("unexpected run error {e:?}"))?
    }
}

#[async_trait]
impl<C: Collection> WithStartAndShutdown for Dht<C> {
    fn is_running(&self) -> bool {
        *self.is_running.lock().unwrap()
    }

    async fn start(&self) {
        let public_key = self.network_secret_key.to_pk();
        let (table_tx, table_rx) = mpsc::channel(self.buffer_size);
        let (task_tx, task_rx) = mpsc::channel(self.buffer_size);

        tokio::spawn(table::server::start_worker::<C>(
            table_rx,
            public_key,
            task_tx.clone(),
            self.local_rep_query.clone(),
            self.shutdown_notify.clone(),
        ));

        let (store_tx, store_rx) = mpsc::channel(self.buffer_size);
        tokio::spawn(store::start_worker(store_rx, self.shutdown_notify.clone()));

        let socket = UdpSocket::bind(self.address)
            .await
            .map(Arc::new)
            .expect("Binding to socket failed");
        tracing::info!("UDP socket bound to {:?}", socket.local_addr().unwrap());

        tokio::spawn(api::start_worker(
            self.socket_rx.lock().unwrap().take().unwrap(),
            task_tx.clone(),
            self.bootstrap_rx.lock().unwrap().take().unwrap(),
            self.shutdown_notify.clone(),
            self.network_secret_key.to_pk(),
            socket.clone(),
        ));

        let bootstrapper = Bootstrapper::new(
            task_tx.clone(),
            table_tx.clone(),
            public_key.0,
            self.nodes.lock().unwrap().take().unwrap_or_default(),
        );

        let (network_event_tx, network_event_rx) = mpsc::channel(self.buffer_size);

        tokio::spawn(network::start_worker(
            network_event_tx,
            table_tx.clone(),
            store_tx.clone(),
            socket.clone(),
            public_key,
            self.shutdown_notify.clone(),
        ));

        tokio::spawn(task::start_worker::<C>(
            task_rx,
            task_tx,
            network_event_rx,
            table_tx.clone(),
            self.shutdown_notify.clone(),
            socket.clone(),
            public_key,
            bootstrapper,
            self.rep_reporter.clone(),
        ));

        if let Err(e) = self.bootstrap().await {
            tracing::error!("DHT failed to bootstrap: {e:?}");
        }

        *self.is_running.lock().unwrap() = true;
    }

    async fn shutdown(&self) {
        self.shutdown_notify.notify_waiters();
        *self.is_running.lock().unwrap() = false;
    }
}

#[async_trait]
impl<C: Collection> DhtInterface<C> for Dht<C> {
    fn init(
        signer: &c![C::SignerInterface],
        _: c![C::TopologyInterface],
        rep_reporter: c!(C::ReputationAggregatorInterface::ReputationReporter),
        local_rep_query: c![C::ReputationAggregatorInterface::ReputationQuery],
        config: Self::Config,
    ) -> Result<Self> {
        let (_, node_public_key) = signer.get_sk();
        Builder::new(node_public_key, config, rep_reporter, local_rep_query).build()
    }

    fn get_socket(&self) -> DhtSocket {
        self.socket.clone()
    }
}

impl<C: Collection> ConfigConsumer for Dht<C> {
    const KEY: &'static str = "dht";

    type Config = Config;
}
