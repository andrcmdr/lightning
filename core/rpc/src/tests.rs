use std::net::SocketAddr;
use std::time::Duration;

use affair::{Executor, TokioSpawn, Worker};
use anyhow::Result;
use fleek_crypto::{
    AccountOwnerSecretKey,
    ConsensusSecretKey,
    EthAddress,
    NodePublicKey,
    NodeSecretKey,
    SecretKey,
};
use hp_fixed::unsigned::HpUfixed;
use jsonrpsee::http_client::{HttpClient, HttpClientBuilder};
use lightning_application::app::Application;
use lightning_application::config::{Config as AppConfig, Mode, StorageConfig};
use lightning_application::genesis::{Genesis, GenesisAccount, GenesisNode};
use lightning_application::query_runner::QueryRunner;
use lightning_blockstore::blockstore::Blockstore;
use lightning_blockstore::config::Config as BlockstoreConfig;
use lightning_blockstore_server::{BlockStoreServer, Config as BlockServerConfig};
use lightning_fetcher::config::Config as FetcherConfig;
use lightning_fetcher::fetcher::Fetcher;
use lightning_indexer::Indexer;
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::{
    Blake3Hash,
    EpochInfo,
    Metadata,
    NodeInfo,
    NodePorts,
    NodeServed,
    ProtocolParams,
    Staking,
    TotalServed,
    TransactionRequest,
    Value,
};
use lightning_interfaces::{
    partial,
    ApplicationInterface,
    BlockStoreInterface,
    BlockStoreServerInterface,
    FetcherInterface,
    IndexerInterface,
    MempoolSocket,
    NotifierInterface,
    OriginProviderInterface,
    PagingParams,
    PoolInterface,
    ReputationAggregatorInterface,
    RpcInterface,
    SignerInterface,
    SyncQueryRunnerInterface,
    WithStartAndShutdown,
};
use lightning_notifier::Notifier;
use lightning_origin_demuxer::OriginDemuxer;
use lightning_pool::{muxer, Config as PoolConfig, PoolProvider};
use lightning_rep_collector::ReputationAggregator;
use lightning_signer::{Config as SignerConfig, Signer};
use lightning_types::Event;
use lightning_utils::application::QueryRunnerExt;
use lightning_utils::rpc as utils;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::FleekApiClient;
use crate::config::Config as RpcConfig;
use crate::Rpc;

#[derive(Serialize, Deserialize, Debug)]
struct RpcSuccessResponse<T> {
    jsonrpc: String,
    id: usize,
    result: T,
}

/// get mempool socket for test cases that do not require consensus
#[derive(Default)]
pub struct MockWorker;

impl MockWorker {
    fn mempool_socket() -> MempoolSocket {
        TokioSpawn::spawn(MockWorker)
    }
}

impl Worker for MockWorker {
    type Request = TransactionRequest;
    type Response = ();

    fn handle(&mut self, _req: Self::Request) -> Self::Response {}
}

partial!(TestBinding {
    ApplicationInterface = Application<Self>;
    FetcherInterface = Fetcher<Self>;
    RpcInterface = Rpc<Self>;
    BlockStoreInterface = Blockstore<Self>;
    BlockStoreServerInterface = BlockStoreServer<Self>;
    OriginProviderInterface = OriginDemuxer<Self>;
    SignerInterface = Signer<Self>;
    NotifierInterface = Notifier<Self>;
    PoolInterface = PoolProvider<Self>;
    ReputationAggregatorInterface = ReputationAggregator<Self>;
    IndexerInterface = Indexer<Self>;
});

struct AppState {
    blockstore: Blockstore<TestBinding>,
}

fn init_rpc(app: Application<TestBinding>, port: u16) -> Result<(Rpc<TestBinding>, AppState)> {
    let mut blockstore = Blockstore::<TestBinding>::init(BlockstoreConfig::default()).unwrap();

    let ipfs_origin =
        OriginDemuxer::<TestBinding>::init(Default::default(), blockstore.clone()).unwrap();

    let signer = Signer::<TestBinding>::init(SignerConfig::test(), app.sync_query()).unwrap();

    let indexer =
        Indexer::<TestBinding>::init(Default::default(), app.sync_query(), &signer).unwrap();

    blockstore.provide_indexer(indexer);

    let notifier = Notifier::<TestBinding>::init(&app);
    let rep_aggregator = ReputationAggregator::<TestBinding>::init(
        Default::default(),
        signer.get_socket(),
        notifier.clone(),
        app.sync_query(),
    )
    .unwrap();

    let pool = PoolProvider::<TestBinding, muxer::quinn::QuinnMuxer>::init(
        PoolConfig::default(),
        &signer,
        app.sync_query(),
        notifier,
        Default::default(),
        rep_aggregator.get_reporter(),
    )
    .unwrap();

    let blockstore_server = BlockStoreServer::<TestBinding>::init(
        BlockServerConfig::default(),
        blockstore.clone(),
        &pool,
        rep_aggregator.get_reporter(),
    )
    .unwrap();

    let fetcher = Fetcher::<TestBinding>::init(
        FetcherConfig::default(),
        blockstore.clone(),
        &blockstore_server,
        Default::default(),
        &ipfs_origin,
    )
    .unwrap();

    let rpc = Rpc::<TestBinding>::init(
        RpcConfig::default_with_port(port),
        MockWorker::mempool_socket(),
        app.sync_query(),
        blockstore.clone(),
        &fetcher,
        &signer,
        None,
    )?;

    Ok((rpc, AppState { blockstore }))
}

async fn init_rpc_without_consensus(
    genesis: Option<Genesis>,
    port: u16,
) -> Result<(Rpc<TestBinding>, QueryRunner)> {
    let blockstore = Blockstore::<TestBinding>::init(BlockstoreConfig::default()).unwrap();
    let app = match genesis {
        Some(genesis) => Application::<TestBinding>::init(
            AppConfig {
                genesis: Some(genesis),
                mode: Mode::Test,
                testnet: false,
                storage: StorageConfig::InMemory,
                db_path: None,
                db_options: None,
            },
            blockstore,
        )
        .unwrap(),
        None => Application::<TestBinding>::init(AppConfig::test(), blockstore).unwrap(),
    };

    let query_runner = app.sync_query();
    app.start().await;

    let (rpc, _) = init_rpc(app, port).unwrap();
    Ok((rpc, query_runner))
}

async fn init_rpc_app_test(port: u16) -> Result<(Rpc<TestBinding>, QueryRunner)> {
    let blockstore = Blockstore::<TestBinding>::init(BlockstoreConfig::default()).unwrap();
    let app = Application::<TestBinding>::init(AppConfig::test(), blockstore).unwrap();
    let query_runner = app.sync_query();
    app.start().await;

    // Init rpc service
    let (rpc, _) = init_rpc(app, port).unwrap();

    Ok((rpc, query_runner))
}

async fn init_admin_rpc_app_test(port: u16) -> Result<(Rpc<TestBinding>, AppState)> {
    let blockstore = Blockstore::<TestBinding>::init(BlockstoreConfig::default()).unwrap();
    let app = Application::<TestBinding>::init(AppConfig::test(), blockstore).unwrap();
    app.start().await;

    init_rpc(app, port)
}

async fn wait_for_server_start(port: u16) -> Result<()> {
    let client = Client::new();
    let mut retries = 10; // Maximum number of retries

    while retries > 0 {
        let response = client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await;
        match response {
            Ok(res) => {
                if res.status().is_success() {
                    println!("Server is ready");
                    break;
                } else {
                    println!(
                        "Server is not ready yet status: {}, res: {}",
                        res.status(),
                        res.text().await?
                    );
                    retries -= 1;
                    // Delay between retries
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            },
            Err(e) => {
                println!("Server Error: {}", e);
                retries -= 1;
                // Delay between retries
                tokio::time::sleep(Duration::from_secs(1)).await;
            },
        }
    }

    if retries > 0 {
        Ok(())
    } else {
        panic!("Server did not become ready within the specified time");
    }
}

fn client(url: SocketAddr) -> HttpClient {
    HttpClientBuilder::default()
        .build(format!("http://{}", url))
        .expect("client to build")
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_ping() -> Result<()> {
    let port = 30000;
    let (rpc, _) = init_rpc_without_consensus(None, port).await.unwrap();
    rpc.start().await;

    let handle = rpc.handle.lock().await;

    assert!(
        !handle
            .as_ref()
            .expect("RPC server to be there")
            .is_stopped()
    );

    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_ping",
        "params":[],
        "id":1,
    });

    let response =
        utils::make_request(format!("http://127.0.0.1:{port}/AHHHHHH"), req.to_string()).await?;

    if response.status().is_success() {
        let response_body = response.text().await?;
        println!("Response body: {response_body}");
    } else {
        panic!("Request failed with status: {}", response.status());
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_flk_balance() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address: EthAddress = owner_public_key.into();

    // Init application service
    let mut genesis = Genesis::load().unwrap();
    genesis.account.push(GenesisAccount {
        public_key: owner_public_key.into(),
        flk_balance: 1000u64.into(),
        stables_balance: 0,
        bandwidth_balance: 0,
    });

    let port = 30001;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_flk_balance",
        "params": {"public_key": eth_address},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<HpUfixed<18>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(HpUfixed::<18>::from(1_000_u32), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_reputation() -> Result<()> {
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    let mut genesis = Genesis::load().unwrap();

    let mut genesis_node = GenesisNode::new(
        owner_public_key.into(),
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            // not used in TestBinding, so defaults are fine
            handshake: Default::default(),
        },
        None,
        true,
    );
    // Init application service and store reputation score in application state.
    genesis_node.reputation = Some(46);
    genesis.node_info.push(genesis_node);
    let port = 30002;
    let (rpc, _query_runner) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_reputation",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<Option<u8>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(Some(46), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_staked() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: 1_000_u32.into(),
        stake_locked_until: 365,
        locked: 0_u32.into(),
        locked_until: 0,
    };

    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 38000,
            worker: 38101,
            mempool: 38102,
            rpc: 38103,
            pool: 38104,
            pinger: 38106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info);

    let port = 30003;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;
    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_staked",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<HpUfixed<18>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(HpUfixed::<18>::from(1_000_u32), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_stables_balance() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address: EthAddress = owner_public_key.into();

    // Init application service
    let mut genesis = Genesis::load().unwrap();
    genesis.account.push(GenesisAccount {
        public_key: owner_public_key.into(),
        flk_balance: 0u64.into(),
        stables_balance: 200,
        bandwidth_balance: 0,
    });

    let port = 30004;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_stables_balance",
        "params": {"public_key": eth_address},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<HpUfixed<6>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(HpUfixed::<6>::from(2_00_u32), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_stake_locked_until() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: 1_000_u32.into(),
        stake_locked_until: 365,
        locked: 0_u32.into(),
        locked_until: 0,
    };
    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info);

    let port = 30005;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_stake_locked_until",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<u64>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(365, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_locked_time() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: 1_000_u32.into(),
        stake_locked_until: 365,
        locked: 0_u32.into(),
        locked_until: 2,
    };
    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info);

    let port = 30006;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_locked_time",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<u64>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(2, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_locked() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: 1_000_u32.into(),
        stake_locked_until: 365,
        locked: 500_u32.into(),
        locked_until: 2,
    };
    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info);

    let port = 30007;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let client = client(rpc.config.addr());

    let res = crate::api::FleekApiClient::get_locked(&client, node_public_key, None).await?;
    assert_eq!(HpUfixed::<18>::from(500_u32), res);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_bandwidth_balance() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address: EthAddress = owner_public_key.into();

    // Init application service
    let mut genesis = Genesis::load().unwrap();
    genesis.account.push(GenesisAccount {
        public_key: owner_public_key.into(),
        flk_balance: 0u64.into(),
        stables_balance: 0,
        bandwidth_balance: 10_000,
    });

    let port = 30008;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_bandwidth_balance",
        "params": {"public_key": eth_address},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<u128>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(10_000, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_node_info() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: 1_000_u32.into(),
        stake_locked_until: 365,
        locked: 500_u32.into(),
        locked_until: 2,
    };
    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info.clone());

    let port = 30009;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_node_info",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<Option<NodeInfo>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(Some(NodeInfo::from(&node_info)), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_staking_amount() -> Result<()> {
    let port = 30010;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_staking_amount",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<u128>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(query_runner.get_staking_amount(), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_committee_members() -> Result<()> {
    let port = 30011;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_committee_members",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<Vec<NodePublicKey>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(query_runner.get_committee_members(), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_epoch() -> Result<()> {
    let port = 30012;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_epoch",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<u64>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(query_runner.get_current_epoch(), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_epoch_info() -> Result<()> {
    let port = 30013;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_epoch_info",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<EpochInfo>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(query_runner.get_epoch_info(), response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_total_supply() -> Result<()> {
    let port = 30014;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_total_supply",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<HpUfixed<18>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;

    let total_supply = match query_runner.get_metadata(&Metadata::TotalSupply) {
        Some(Value::HpUfixed(s)) => s,
        _ => panic!("TotalSupply is set genesis and should never be empty"),
    };

    assert_eq!(total_supply, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_year_start_supply() -> Result<()> {
    let port = 30015;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_year_start_supply",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<HpUfixed<18>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;

    let supply_year_start = match query_runner.get_metadata(&Metadata::SupplyYearStart) {
        Some(Value::HpUfixed(s)) => s,
        _ => panic!("SupplyYearStart is set genesis and should never be empty"),
    };

    assert_eq!(supply_year_start, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_protocol_fund_address() -> Result<()> {
    let port = 30016;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_protocol_fund_address",
        "params":[],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<EthAddress>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;

    let protocol_account = match query_runner.get_metadata(&Metadata::ProtocolFundAddress) {
        Some(Value::AccountPublicKey(s)) => s,
        _ => panic!("AccountPublicKey is set genesis and should never be empty"),
    };

    assert_eq!(protocol_account, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_protocol_params() -> Result<()> {
    let port = 30017;
    let (rpc, query_runner) = init_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let params = ProtocolParams::LockTime;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_protocol_params",
        "params": {"protocol_params": params},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<u128>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(
        query_runner.get_protocol_param(&params).unwrap(),
        response.result
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_total_served() -> Result<()> {
    // Init application service and store total served in application state.
    let mut genesis = Genesis::load().unwrap();

    let total_served = TotalServed {
        served: vec![1000],
        reward_pool: 1_000_u32.into(),
    };
    genesis.total_served.insert(0, total_served.clone());

    let port = 30018;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_total_served",
        "params": { "epoch": 0 },
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<TotalServed>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(total_served, response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_node_served() -> Result<()> {
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store total served in application state.
    let mut genesis = Genesis::load().unwrap();
    let mut genesis_node = GenesisNode::new(
        owner_public_key.into(),
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        None,
        true,
    );

    genesis_node.current_epoch_served = Some(NodeServed {
        served: vec![1000],
        ..Default::default()
    });
    genesis.node_info.push(genesis_node);

    let port = 30019;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_node_served",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<NodeServed>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(vec![1000], response.result.served);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_is_valid_node() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: genesis.min_stake.into(),
        stake_locked_until: 0,
        locked: 0_u32.into(),
        locked_until: 0,
    };
    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info);

    let port = 30020;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_is_valid_node",
        "params": {"public_key": node_public_key},
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<bool>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert!(response.result);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_get_node_registry() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();
    let eth_address = owner_public_key.into();
    let node_secret_key = NodeSecretKey::generate();
    let node_public_key = node_secret_key.to_pk();
    let consensus_secret_key = ConsensusSecretKey::generate();
    let consensus_public_key = consensus_secret_key.to_pk();

    // Init application service and store node info in application state.
    let mut genesis = Genesis::load().unwrap();
    let staking = Staking {
        staked: genesis.min_stake.into(),
        stake_locked_until: 0,
        locked: 0_u32.into(),
        locked_until: 0,
    };
    let node_info = GenesisNode::new(
        eth_address,
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 48000,
            worker: 48101,
            mempool: 48102,
            rpc: 48103,
            pool: 48104,
            pinger: 48106,
            handshake: Default::default(),
        },
        Some(staking),
        false,
    );

    genesis.node_info.push(node_info.clone());

    let committee_size =
        genesis.node_info.iter().fold(
            0,
            |acc, node| {
                if node.genesis_committee { acc + 1 } else { acc }
            },
        );

    let port = 30021;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_node_registry",
        "params": [],
        "id":1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<Vec<NodeInfo>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert_eq!(response.result.len(), committee_size + 1);
    assert!(response.result.contains(&NodeInfo::from(&node_info)));

    let req = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_node_registry",
        "params": PagingParams{ ignore_stake: true, start: committee_size as u32, limit: 10 },
        "id":1,
    });

    let response = utils::rpc_request::<Vec<NodeInfo>>(
        &client,
        format!("http://127.0.0.1:{port}/rpc/v0"),
        req.to_string(),
    )
    .await?;
    assert!(response.result.contains(&NodeInfo::from(&node_info)));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_admin_rpc_store() -> Result<()> {
    let port = 30022;
    let (rpc, app) = init_admin_rpc_app_test(port).await.unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let req = json!({
        "jsonrpc": "2.0",
        "method": "admin_store",
        "params": { "path": "../test-utils/files/index.ts" },
        "id": 1,
    });

    let client = Client::new();
    let response = utils::rpc_request::<Blake3Hash>(
        &client,
        format!("http://127.0.0.1:{port}/admin"),
        req.to_string(),
    )
    .await?;

    let expected_content: Vec<u8> = std::fs::read("../test-utils/files/index.ts").unwrap();
    let stored_content = app
        .blockstore
        .read_all_to_vec(&response.result)
        .await
        .unwrap();
    assert_eq!(expected_content, stored_content);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
// #[traced_test]
async fn test_rpc_events() -> Result<()> {
    // Create keys
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let owner_public_key = owner_secret_key.to_pk();

    // Init application service
    let mut genesis = Genesis::load().unwrap();
    genesis.account.push(GenesisAccount {
        public_key: owner_public_key.into(),
        flk_balance: 1000u64.into(),
        stables_balance: 0,
        bandwidth_balance: 0,
    });

    let port = 30023;
    let (rpc, _) = init_rpc_without_consensus(Some(genesis), port)
        .await
        .unwrap();

    rpc.start().await;
    wait_for_server_start(port).await?;

    let sender = rpc.event_tx();

    let client = jsonrpsee::ws_client::WsClientBuilder::default()
        .build(&format!("ws://127.0.0.1:{port}/rpc/v0"))
        .await?;

    let mut sub = FleekApiClient::handle_subscription(&client, None).await?;

    let event = Event::transfer(
        EthAddress::from([0; 20]),
        EthAddress::from([1; 20]),
        EthAddress::from([2; 20]),
        HpUfixed::<18>::from(10_u16),
    );

    sender
        .send(vec![event.clone()])
        .await
        .expect("can send event");

    assert_eq!(sub.next().await.expect("An event from the sub")?, event);

    Ok(())
}
