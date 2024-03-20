use std::collections::HashSet;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use bytes::Bytes;
use fleek_crypto::{
    AccountOwnerSecretKey,
    ConsensusSecretKey,
    NodePublicKey,
    NodeSecretKey,
    SecretKey,
};
use futures::StreamExt;
use lightning_application::app::Application;
use lightning_application::config::{Config as AppConfig, Mode, StorageConfig};
use lightning_application::genesis::{Genesis, GenesisNode};
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::{NodeIndex, NodePorts};
use lightning_interfaces::{
    partial,
    ApplicationInterface,
    EventHandlerInterface,
    NotifierInterface,
    PoolInterface,
    ReputationAggregatorInterface,
    RequestInterface,
    RequesterInterface,
    ResponderInterface,
    ResponseInterface,
    ServiceScope,
    SignerInterface,
    SyncQueryRunnerInterface,
    TopologyInterface,
    WithStartAndShutdown,
};
use lightning_notifier::Notifier;
use lightning_rep_collector::ReputationAggregator;
use lightning_signer::{utils, Config as SignerConfig, Signer};
use lightning_topology::{Config as TopologyConfig, Topology};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::endpoint::EndpointTask;
use crate::event::{Event, EventReceiver, Param};
use crate::{muxer, provider, Config, PoolProvider};
partial!(TestBinding {
    ApplicationInterface = Application<Self>;
    PoolInterface = PoolProvider<Self>;
    SignerInterface = Signer<Self>;
    NotifierInterface = Notifier<Self>;
    TopologyInterface = Topology<Self>;
    ReputationAggregatorInterface = ReputationAggregator<Self>;
});

pub struct Peer<C: Collection> {
    // We hold on to the rep aggregator and notifier so
    // that they do not get dropped and cause a
    // race condition which causes the pool to stop.
    _rep_aggregator: C::ReputationAggregatorInterface,
    _notifier: C::NotifierInterface,
    pool: C::PoolInterface,
    signer: Signer<C>,
    topology: C::TopologyInterface,
    pub node_public_key: NodePublicKey,
    pub node_index: NodeIndex,
}

struct PathBuffWrapper(PathBuf);

impl Drop for PathBuffWrapper {
    fn drop(&mut self) {
        if self.0.exists() {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
}

async fn get_pools(
    test_name: &str,
    port_offset: u16,
    num_peers: usize,
    state_server_address_port: Option<u16>,
) -> (
    Vec<Peer<TestBinding>>,
    Application<TestBinding>,
    PathBuffWrapper,
) {
    let mut signers_configs = Vec::new();
    let mut genesis = Genesis::load().unwrap();
    let path = std::env::temp_dir()
        .join("lightning-pool-test")
        .join(test_name);
    if path.exists() {
        std::fs::remove_dir_all(&path).unwrap();
    }
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();

    genesis.node_info = vec![];

    // Create signer configs and add nodes to state.
    for i in 0..num_peers {
        let node_secret_key = NodeSecretKey::generate();
        let consensus_secret_key = ConsensusSecretKey::generate();
        let node_key_path = path.join(format!("node{i}/node.pem"));
        let consensus_key_path = path.join(format!("node{i}/cons.pem"));
        utils::save(&node_key_path, node_secret_key.encode_pem()).unwrap();
        utils::save(&consensus_key_path, consensus_secret_key.encode_pem()).unwrap();
        let signer_config = SignerConfig {
            node_key_path: node_key_path.try_into().unwrap(),
            consensus_key_path: consensus_key_path.try_into().unwrap(),
        };

        signers_configs.push(signer_config);

        genesis.node_info.push(GenesisNode::new(
            owner_public_key.into(),
            node_secret_key.to_pk(),
            "127.0.0.1".parse().unwrap(),
            consensus_secret_key.to_pk(),
            "127.0.0.1".parse().unwrap(),
            node_secret_key.to_pk(),
            NodePorts {
                primary: 48000_u16,
                worker: 48101_u16,
                mempool: 48202_u16,
                rpc: 48300_u16,
                pool: port_offset + i as u16,
                pinger: 48600_u16,
                // Handshake is unused so the defaults are fine.
                handshake: Default::default(),
            },
            None,
            true,
        ));
    }

    let app = Application::<TestBinding>::init(
        AppConfig {
            genesis: Some(genesis),
            genesis_path: None,
            mode: Mode::Test,
            testnet: false,
            storage: StorageConfig::InMemory,
            db_path: None,
            db_options: None,
        },
        Default::default(),
    )
    .unwrap();
    app.start().await;

    // Create peers.
    let mut peers = Vec::new();
    for (i, signer_config) in signers_configs.into_iter().enumerate() {
        let address: SocketAddr = format!("0.0.0.0:{}", port_offset + i as u16)
            .parse()
            .unwrap();
        let peer = create_peer(
            &app,
            signer_config,
            address,
            true,
            state_server_address_port.map(|port| port + i as u16),
        );
        peers.push(peer);
    }

    (peers, app, PathBuffWrapper(path))
}

// Create a peer that is not in state.
fn create_unknown_peer(
    path: PathBuf,
    app: &Application<TestBinding>,
    peer_id: usize,
    address: SocketAddr,
) -> Peer<TestBinding> {
    let node_secret_key = NodeSecretKey::generate();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let node_key_path = path.join(format!("node{peer_id}/node.pem"));
    let consensus_key_path = path.join(format!("node{peer_id}/cons.pem"));
    utils::save(&node_key_path, node_secret_key.encode_pem()).unwrap();
    utils::save(&consensus_key_path, consensus_secret_key.encode_pem()).unwrap();
    let signer_config = SignerConfig {
        node_key_path: node_key_path.try_into().unwrap(),
        consensus_key_path: consensus_key_path.try_into().unwrap(),
    };
    create_peer(app, signer_config, address, false, None)
}

fn create_peer(
    app: &Application<TestBinding>,
    signer_config: SignerConfig,
    address: SocketAddr,
    in_state: bool,
    state_server_address_port: Option<u16>,
) -> Peer<TestBinding> {
    let query_runner = app.sync_query();
    let signer = Signer::<TestBinding>::init(signer_config, query_runner.clone()).unwrap();
    let notifier = Notifier::<TestBinding>::init(app);
    let topology = Topology::<TestBinding>::init(
        TopologyConfig::default(),
        signer.get_ed25519_pk(),
        query_runner.clone(),
    )
    .unwrap();
    let rep_aggregator = ReputationAggregator::<TestBinding>::init(
        Default::default(),
        signer.get_socket(),
        notifier.clone(),
        query_runner.clone(),
    )
    .unwrap();

    let http = state_server_address_port
        .map(|port| SocketAddr::from((IpAddr::from([127, 0, 0, 1]), port)));

    let config = Config {
        max_idle_timeout: Duration::from_secs(5),
        address,
        http,
    };
    let pool = PoolProvider::<TestBinding, muxer::quinn::QuinnMuxer>::init(
        config,
        &signer,
        query_runner.clone(),
        notifier.clone(),
        topology.clone(),
        rep_aggregator.get_reporter(),
    )
    .unwrap();

    let node_public_key = signer.get_ed25519_pk();
    let node_index = if in_state {
        query_runner.pubkey_to_index(&node_public_key).unwrap()
    } else {
        u32::MAX
    };

    Peer::<TestBinding> {
        _rep_aggregator: rep_aggregator,
        _notifier: notifier,
        signer,
        topology,
        pool,
        node_public_key,
        node_index,
    }
}

struct EventReceiverTestState {
    event_tx: Sender<Event>,
    endpoint_task_rx: Receiver<EndpointTask>,
    shutdown: CancellationToken,
    _notifier: Notifier<TestBinding>,
}

fn event_receiver(
    app: Application<TestBinding>,
    peer: &Peer<TestBinding>,
) -> (EventReceiver<TestBinding>, EventReceiverTestState) {
    let query_runner = app.sync_query();
    let topology = peer.topology.clone();
    let notifier = peer._notifier.clone();
    let (_, sk) = peer.signer.get_sk();
    let pk = sk.to_pk();

    let (event_tx, event_rx) = mpsc::channel(8);
    let (endpoint_task_tx, endpoint_task_rx) = mpsc::channel(8);

    let shutdown = CancellationToken::new();

    (
        EventReceiver::<TestBinding>::new(
            query_runner,
            topology,
            notifier.clone(),
            event_rx,
            endpoint_task_tx,
            pk,
            shutdown.clone(),
        ),
        EventReceiverTestState {
            event_tx,
            endpoint_task_rx,
            _notifier: notifier,
            shutdown,
        },
    )
}

#[tokio::test]
async fn test_send_to_one() {
    // Given: two peers.
    let (peers, app, path) = get_pools("send_to_one", 48000, 2, None).await;
    let query_runner = app.sync_query();

    let node_index1 = query_runner
        .pubkey_to_index(&peers[0].node_public_key)
        .unwrap();
    let node_index2 = query_runner
        .pubkey_to_index(&peers[1].node_public_key)
        .unwrap();

    let event_handler1 = peers[0].pool.open_event(ServiceScope::Broadcast);
    let mut event_handler2 = peers[1].pool.open_event(ServiceScope::Broadcast);

    for peer in &peers {
        peer.pool.start().await;
    }

    // When: one of the peers sends a message to the other peer.
    let msg = Bytes::from("hello");
    event_handler1.send_to_one(node_index2, msg.clone());

    // Then: the other peer receives the message.
    let (sender, recv_msg) = event_handler2.receive().await.unwrap();
    assert_eq!(recv_msg, msg);
    assert_eq!(sender, node_index1);

    // Clean up.
    for peer in &peers {
        peer.pool.shutdown().await;
    }
    drop(path);
}

#[tokio::test]
async fn test_send_to_all() {
    // Given: a list of peers that are in state and some that are not.
    let port_offset = 49000;
    let (peers, app, path) = get_pools("send_to_all", port_offset, 4, None).await;
    let unknown_peer = create_unknown_peer(
        path.0.clone(),
        &app,
        peers.len(),
        format!("0.0.0.0:{}", port_offset + peers.len() as u16)
            .parse()
            .unwrap(),
    );
    let query_runner = app.sync_query();

    // Given: we start known nodes.
    let node_index1 = query_runner
        .pubkey_to_index(&peers[0].node_public_key)
        .unwrap();
    let mut event_handlers: Vec<_> = peers
        .iter()
        .map(|peer| peer.pool.open_event(ServiceScope::Broadcast))
        .collect();
    for peer in &peers {
        peer.pool.start().await;
    }

    // Given: we start an unknown node.
    let mut event_handlers_unknown_peer = unknown_peer.pool.open_event(ServiceScope::Broadcast);

    unknown_peer.pool.start().await;

    // When: one of the nodes sends a message.
    let msg = Bytes::from("hello");
    event_handlers[0].send_to_all(msg.clone(), |_| true);

    // Then: the nodes in the network receive the message.
    #[allow(clippy::needless_range_loop)]
    for i in 1..peers.len() {
        let (sender, recv_msg) = event_handlers[i].receive().await.unwrap();
        assert_eq!(recv_msg, msg);
        assert_eq!(sender, node_index1);
    }

    // Then: the unknown node does not receive the message.
    assert!(
        tokio::time::timeout(
            Duration::from_secs(5),
            event_handlers_unknown_peer.receive(),
        )
        .await
        .is_err()
    );

    // Clean up.
    for peer in &peers {
        peer.pool.shutdown().await;
    }
    unknown_peer.pool.shutdown().await;
    drop(path);
}

#[tokio::test]
async fn test_open_req_res() {
    // Given: two peers.
    let (peers, app, path) = get_pools("open_req_res", 50000, 2, None).await;
    let query_runner = app.sync_query();

    let node_index1 = query_runner
        .pubkey_to_index(&peers[0].node_public_key)
        .unwrap();
    let node_index2 = query_runner
        .pubkey_to_index(&peers[1].node_public_key)
        .unwrap();
    let (_requester1, mut responder1) = peers[0].pool.open_req_res(ServiceScope::BlockstoreServer);
    let (requester2, _responder2) = peers[1].pool.open_req_res(ServiceScope::BlockstoreServer);

    for peer in &peers {
        peer.pool.start().await;
    }

    // When: one of the peer sends a request to the other peer.
    let chunks = vec![
        Bytes::from("one"),
        Bytes::from("two"),
        Bytes::from("three"),
        Bytes::from("end"),
    ];
    let chunks_clone = chunks.clone();

    // Then: We receive the response.
    let sender_fut = async move {
        let (request_header, mut request) = responder1.get_next_request().await.unwrap();
        assert_eq!(request_header.peer, node_index2);
        assert_eq!(request_header.bytes, Bytes::from("a hash"));
        for chunk in chunks_clone {
            request.send(chunk).await.unwrap();
        }
    };

    // The peer that receives the request sends a response.
    let recv_fut = async move {
        let end_marker = chunks[chunks.len() - 1].clone();
        let response = requester2
            .request(node_index1, Bytes::from("a hash"))
            .await
            .unwrap();
        response.status_code().unwrap();
        let mut body = response.body();
        let mut i = 0;
        loop {
            let chunk = body.next().await.unwrap().unwrap();
            assert_eq!(chunk, chunks[i]);
            if chunk == end_marker {
                break;
            }
            i += 1;
        }
    };

    futures::join!(sender_fut, recv_fut);

    // Clean up.
    for peer in &peers {
        peer.pool.shutdown().await;
    }
    drop(path);
}

#[tokio::test]
async fn test_open_req_res_unknown_peer() {
    // Give: a peer.
    let (peers, _app, path) = get_pools("test_open_req_res_unknown_peer", 55000, 1, None).await;
    let (requester1, _responder1) = peers[0].pool.open_req_res(ServiceScope::BlockstoreServer);
    peers[0].pool.start().await;

    // Given: an index for an unknown node.
    let unknown_index = 6969;

    // When: we send a request to a node that is not in state.
    let response = requester1
        .request(unknown_index, Bytes::from("a hasshh"))
        .await;

    // Then: our request fails.
    let _expected_err: Result<provider::Response, io::Error> =
        Err(io::Error::from(io::ErrorKind::AddrNotAvailable));
    assert!(matches!(response, _expected_err));

    // Clean up.
    peers[0].pool.shutdown().await;
    drop(path);
}

#[tokio::test]
async fn test_log_pool_get_index() {
    // We never bind.
    let (peers, app, _path) = get_pools("test_log_pool_get_index", 8000, 2, None).await;
    let (event_receiver, _) = event_receiver(app, &peers[0]);
    assert_eq!(event_receiver.handler.get_index(), peers[0].node_index);
}

#[tokio::test]
async fn test_log_pool_update_connections() {
    // Given: a network of 4 nodes.
    // We never bind.
    let (peers, app, _path) = get_pools("test_log_pool_update_connections", 8000, 4, None).await;
    let (mut event_receiver, _state) = event_receiver(app, &peers[0]);

    // When: we tell first node to connect to all the peers.
    // Skip the first one.
    let peers_to_connect = peers[1..]
        .iter()
        .map(|peer| peer.node_index)
        .collect::<HashSet<_>>();
    event_receiver
        .handler
        ._update_connections(peers_to_connect.clone());

    // Then: state in the pool includes the expected peers.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(peers_to_connect, connections);

    // When: we tell first node to connect to all the peers plus some random peers in state.
    // Skip the first one.
    let mut peers_to_connect = peers[1..]
        .iter()
        .map(|peer| peer.node_index)
        .collect::<HashSet<_>>();
    peers_to_connect.insert(6969);
    peers_to_connect.insert(9696);
    event_receiver
        .handler
        ._update_connections(peers_to_connect.clone());

    // Then: state in the pool includes only the peers that are in state.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(
        connections,
        peers[1..]
            .iter()
            .map(|peer| peer.node_index)
            .collect::<HashSet<_>>()
    );

    // When: we tell first node to connect to update its state to only one peer.
    // Skip the first one.
    let peers_to_connect = peers[1..2]
        .iter()
        .map(|peer| peer.node_index)
        .collect::<HashSet<_>>();
    let task = event_receiver
        .handler
        ._update_connections(peers_to_connect.clone());

    // Then: Task includes peers to keep and drop.
    let EndpointTask::Update { keep, drop } = task else {
        panic!("invalid task");
    };
    assert!(keep.contains_key(&peers[1].node_index));
    assert_eq!(keep.len(), 1);
    assert!(drop.contains(&peers[2].node_index));
    assert!(drop.contains(&peers[3].node_index));
    assert_eq!(drop.len(), 2);

    // Then: state in the pool includes the expected peers.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(peers_to_connect.len(), 1);
    assert_eq!(peers_to_connect, connections);

    // When: we tell first node to disconnect from all nodes.
    // Skip the first one.
    event_receiver.handler._update_connections(HashSet::new());

    // Then: state in the pool includes the expected peers.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert!(connections.is_empty());
}

#[tokio::test]
async fn test_log_pool_pinning_peers_from_topology() {
    // Given: a network of 4 nodes.
    // We never bind.
    let (peers, app, _path) =
        get_pools("test_log_pool_pinning_peers_from_topology", 8000, 4, None).await;
    let (mut event_receiver, mut state) = event_receiver(app, &peers[0]);

    // Given: a node connects to all.
    let all_peers = peers[1..]
        .iter()
        .map(|peer| peer.node_index)
        .collect::<HashSet<_>>();
    event_receiver.handler._update_connections(all_peers);

    // When: we send a send-request to one peer.
    let pinned_peer_index = peers[1].node_index;
    let (respond, _) = oneshot::channel();
    state
        .event_tx
        .send(Event::SendRequest {
            dst: pinned_peer_index,
            service_scope: ServiceScope::BlockstoreServer,
            request: Bytes::new(),
            respond,
        })
        .await
        .unwrap();

    // Spawn the receiver.
    let handle = event_receiver.spawn();

    // We poll the event receiver to move past the initial set up task.
    let mut task = None;
    while let Some(next_task) = state.endpoint_task_rx.recv().await {
        if let next_task @ EndpointTask::SendRequest { .. } = next_task {
            task = Some(next_task);
            state.shutdown.cancel();
            break;
        }
    }
    let task = task.unwrap();
    assert!(matches!(task, EndpointTask::SendRequest { .. }));

    // We get the receiver back.
    let mut event_receiver = handle.await.unwrap();

    // When: we tell pool to disconnect from all current peers.
    event_receiver.handler._update_connections(HashSet::new());

    // Then: that peer gets pinned and does not get dropped on an update.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    let mut expected_peers = HashSet::new();
    expected_peers.insert(peers[1].node_index);
    assert_eq!(connections.len(), 1);
    assert_eq!(expected_peers, connections);

    // When: we unpin the peer and tell the pool to disconnect.
    event_receiver.handler.clean(peers[1].node_index);

    // Then: the peer is cleared from the pool state.
    event_receiver.handler._update_connections(HashSet::new());
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert!(connections.is_empty());
}

#[tokio::test]
async fn test_log_pool_pinning_peers_outside_topology_cluster() {
    // Given: a network of 2 nodes.
    let (peers, app, _path) = get_pools(
        "test_log_pool_pinning_peers_outside_topology_cluster",
        8000, // We never bind.
        2,
        None,
    )
    .await;
    let (mut event_receiver, _state) = event_receiver(app, &peers[0]);

    // Given: no peers in state.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert!(connections.is_empty());

    // Given: we send a send-request request to the pool and thereby pinning the node.
    // Todo: when we have a mock topology, we can spawn a receiver and send events
    // instead of bypassing the initial set-up which uses topology in a way that
    // we dont want for this test.
    let pinned_peer_index = peers[1].node_index;
    let (respond, _) = oneshot::channel();
    event_receiver
        .handle_event(Event::SendRequest {
            dst: pinned_peer_index,
            service_scope: ServiceScope::BlockstoreServer,
            request: Bytes::new(),
            respond,
        })
        .unwrap();

    // When: we clean the pinned connection.
    event_receiver.handler.clean(peers[1].node_index);

    // Then: Since it wasn't a peer that was supposed to be in our cluster,
    // it gets immediately cleared out from the pool state.
    let connections = event_receiver
        .handler
        .pool
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    assert!(connections.is_empty());
}

#[tokio::test]
async fn test_log_pool_only_broadcast_to_peers_in_topology_cluster() {
    // Given: a network of 4 nodes.
    let (peers, app, _path) = get_pools(
        "test_log_pool_only_broadcast_to_peers_in_topology_cluster",
        8000, // We never bind.
        4,
        None,
    )
    .await;
    let (mut event_receiver, mut state) = event_receiver(app, &peers[0]);

    // Given: we connect to 2 peers.
    let peers_to_connect = peers[1..3]
        .iter()
        .map(|peer| peer.node_index)
        .collect::<HashSet<_>>();
    event_receiver
        .handler
        ._update_connections(peers_to_connect.clone());
    assert_eq!(event_receiver.handler.pool.len(), 2);

    // Given: we send a send-request request to the pool and thereby pinning the node.
    let pinned_peer_index = peers[3].node_index;
    let (respond, _) = oneshot::channel();
    event_receiver
        .handle_event(Event::SendRequest {
            dst: pinned_peer_index,
            service_scope: ServiceScope::BlockstoreServer,
            request: Bytes::new(),
            respond,
        })
        .unwrap();

    assert!(matches!(
        state.endpoint_task_rx.recv().await.unwrap(),
        EndpointTask::SendRequest { .. }
    ));

    // When: we send a broadcast message.
    event_receiver
        .handle_event(Event::Broadcast {
            service_scope: ServiceScope::Broadcast,
            message: Bytes::new(),
            param: Param::Filter(Box::new(|_| true)),
        })
        .unwrap();

    // Then: the task is to broadcast only to peers from topology and not the pinned peer.
    match state.endpoint_task_rx.recv().await.unwrap() {
        EndpointTask::SendMessage {
            peers: peers_to_broadcast,
            ..
        } => {
            let peers_to_broadcast = peers_to_broadcast
                .into_iter()
                .map(|info| info.node_info.index)
                .collect::<HashSet<_>>();
            assert_eq!(peers_to_broadcast, peers_to_connect);
            assert!(!peers_to_broadcast.contains(&peers[3].node_index))
        },
        _ => {
            panic!("invalid value expected a broadcast task")
        },
    }
}

#[tokio::test]
async fn test_log_pool_only_broadcast_to_one_peer() {
    // Given: a network of 4 nodes.
    let (peers, app, _path) = get_pools(
        "test_log_pool_only_broadcast_to_one_peer",
        8000, // We never bind.
        4,
        None,
    )
    .await;
    let (mut event_receiver, mut state) = event_receiver(app, &peers[0]);

    // Given: we connect to all peers.
    let peers_to_connect = peers[1..]
        .iter()
        .map(|peer| peer.node_index)
        .collect::<HashSet<_>>();
    event_receiver
        .handler
        ._update_connections(peers_to_connect.clone());
    assert_eq!(event_receiver.handler.pool.len(), 3);

    // When: we send a broadcast message to one peer.
    event_receiver
        .handle_event(Event::Broadcast {
            service_scope: ServiceScope::Broadcast,
            message: Bytes::new(),
            param: Param::Index(peers[1].node_index),
        })
        .unwrap();

    // Then: the message would be sent to that peer only.
    match state.endpoint_task_rx.recv().await.unwrap() {
        EndpointTask::SendMessage {
            peers: peers_to_broadcast,
            ..
        } => {
            let peers_to_broadcast = peers_to_broadcast
                .into_iter()
                .map(|info| info.node_info.index)
                .collect::<HashSet<_>>();
            assert_eq!(peers_to_broadcast.len(), 1);
            assert!(peers_to_broadcast.contains(&peers[1].node_index))
        },
        _ => {
            unreachable!("invalid value expected a broadcast task")
        },
    }

    // When: we send a broadcast message to a peer that is not in state.
    event_receiver
        .handle_event(Event::Broadcast {
            service_scope: ServiceScope::Broadcast,
            message: Bytes::new(),
            param: Param::Index(6969),
        })
        .unwrap();

    // Then: the broadcast request gets ignored.
    assert!(
        tokio::time::timeout(Duration::from_secs(5), state.endpoint_task_rx.recv())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn test_start_shutdown() {
    // Given: two peers.
    let (peers, app, _path) = get_pools("start_shutdown", 60000, 2, Some(60010)).await;
    let query_runner = app.sync_query();

    let node_index1 = query_runner
        .pubkey_to_index(&peers[0].node_public_key)
        .unwrap();
    let node_index2 = query_runner
        .pubkey_to_index(&peers[1].node_public_key)
        .unwrap();

    let event_handler1 = peers[0].pool.open_event(ServiceScope::Broadcast);
    let mut event_handler2 = peers[1].pool.open_event(ServiceScope::Broadcast);

    // Given: we start the peers.
    for peer in &peers {
        peer.pool.start().await;
    }

    // Given: we exchange data over the network.
    let msg = Bytes::from("hello");
    event_handler1.send_to_one(node_index2, msg.clone());
    let (sender, recv_msg) = event_handler2.receive().await.unwrap();
    assert_eq!(recv_msg, msg);
    assert_eq!(sender, node_index1);

    // When: we shutdown.
    for peer in &peers {
        peer.pool.shutdown().await;
    }

    // Then: we should be able to restart immediately again without issues.
    for peer in &peers {
        peer.pool.start().await;
    }

    // Clean up.
    for peer in &peers {
        peer.pool.shutdown().await;
    }
}
