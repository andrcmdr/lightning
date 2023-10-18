use std::collections::BTreeMap;
use std::time::SystemTime;
use std::vec;

use affair::Socket;
use anyhow::{anyhow, Result};
use fleek_crypto::{
    AccountOwnerSecretKey,
    ConsensusPublicKey,
    ConsensusSecretKey,
    NodePublicKey,
    NodeSecretKey,
    SecretKey,
};
use hp_fixed::unsigned::HpUfixed;
use lightning_interfaces::application::ExecutionEngineSocket;
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::{
    Block,
    BlockExecutionResponse,
    DeliveryAcknowledgment,
    Epoch,
    ExecutionError,
    HandshakePorts,
    NodePorts,
    ProofOfConsensus,
    ProtocolParams,
    Tokens,
    TotalServed,
    TransactionRequest,
    TransactionResponse,
    UpdateMethod,
    UpdatePayload,
    UpdateRequest,
};
use lightning_interfaces::{
    partial,
    ApplicationInterface,
    PagingParams,
    SyncQueryRunnerInterface,
    ToDigest,
};
use lightning_test_utils::{random, reputation};
use tokio::test;

use crate::app::Application;
use crate::config::{Config, Mode, StorageConfig};
use crate::genesis::{Genesis, GenesisNode};
use crate::query_runner::QueryRunner;

partial!(TestBinding {
    ApplicationInterface = Application<Self>;
});

pub struct Params {
    epoch_time: Option<u64>,
    max_inflation: Option<u16>,
    protocol_share: Option<u16>,
    node_share: Option<u16>,
    service_builder_share: Option<u16>,
    max_boost: Option<u16>,
    supply_at_genesis: Option<u64>,
}

#[derive(Clone)]
struct GenesisCommitteeKeystore {
    _owner_secret_key: AccountOwnerSecretKey,
    node_secret_key: NodeSecretKey,
    _consensus_secret_key: ConsensusSecretKey,
    _worker_secret_key: NodeSecretKey,
}

fn get_genesis_committee(num_members: usize) -> (Vec<GenesisNode>, Vec<GenesisCommitteeKeystore>) {
    let mut keystore = Vec::new();
    let mut committee = Vec::new();
    (0..num_members as u16).for_each(|i| {
        let node_secret_key = NodeSecretKey::generate();
        let consensus_secret_key = ConsensusSecretKey::generate();
        let owner_secret_key = AccountOwnerSecretKey::generate();
        add_to_committee(
            &mut committee,
            &mut keystore,
            node_secret_key,
            consensus_secret_key,
            owner_secret_key,
            i,
        )
    });
    (committee, keystore)
}

fn add_to_committee(
    committee: &mut Vec<GenesisNode>,
    keystore: &mut Vec<GenesisCommitteeKeystore>,
    node_secret_key: NodeSecretKey,
    consensus_secret_key: ConsensusSecretKey,
    owner_secret_key: AccountOwnerSecretKey,
    index: u16,
) {
    let node_public_key = node_secret_key.to_pk();
    let consensus_public_key = consensus_secret_key.to_pk();
    let owner_public_key = owner_secret_key.to_pk();
    committee.push(GenesisNode::new(
        owner_public_key.into(),
        node_public_key,
        "127.0.0.1".parse().unwrap(),
        consensus_public_key,
        "127.0.0.1".parse().unwrap(),
        node_public_key,
        NodePorts {
            primary: 8000 + index,
            worker: 9000 + index,
            mempool: 7000 + index,
            rpc: 6000 + index,
            pool: 5000 + index,
            dht: 4000 + index,
            handshake: HandshakePorts {
                http: 5000 + index,
                webrtc: 6000 + index,
                webtransport: 7000 + index,
            },
        },
        None,
        true,
    ));
    keystore.push(GenesisCommitteeKeystore {
        _owner_secret_key: owner_secret_key,
        _worker_secret_key: node_secret_key.clone(),
        node_secret_key,
        _consensus_secret_key: consensus_secret_key,
    });
}

fn get_new_committee(
    query_runner: &QueryRunner,
    committee: &[GenesisNode],
    keystore: &[GenesisCommitteeKeystore],
) -> (Vec<GenesisNode>, Vec<GenesisCommitteeKeystore>) {
    let mut new_committee = Vec::new();
    let mut new_keystore = Vec::new();
    let committee_members = query_runner.get_committee_members();
    for node in committee_members {
        let index = committee
            .iter()
            .enumerate()
            .find_map(|(index, c)| {
                if c.primary_public_key == node {
                    Some(index)
                } else {
                    None
                }
            })
            .expect("Committe member was not found in genesis Committee");
        new_committee.push(committee[index].clone());
        new_keystore.push(keystore[index].clone());
    }
    (new_committee, new_keystore)
}

// Init the app and return the execution engine socket that would go to narwhal and the query socket
// that could go to anyone
fn init_app(config: Option<Config>) -> (ExecutionEngineSocket, QueryRunner) {
    let config = config.or(Some(Config {
        genesis: None,
        mode: Mode::Dev,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));
    let app = Application::<TestBinding>::init(config.unwrap(), Default::default()).unwrap();

    (app.transaction_executor(), app.sync_query())
}

fn init_app_with_params(
    params: Params,
    committee: Option<Vec<GenesisNode>>,
) -> (ExecutionEngineSocket, QueryRunner) {
    let mut genesis = Genesis::load().expect("Failed to load genesis from file.");

    if let Some(committee) = committee {
        genesis.node_info = committee;
    }

    genesis.epoch_start = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    if let Some(epoch_time) = params.epoch_time {
        genesis.epoch_time = epoch_time;
    }

    if let Some(max_inflation) = params.max_inflation {
        genesis.max_inflation = max_inflation;
    }

    if let Some(protocol_share) = params.protocol_share {
        genesis.protocol_share = protocol_share;
    }

    if let Some(node_share) = params.node_share {
        genesis.node_share = node_share;
    }

    if let Some(service_builder_share) = params.service_builder_share {
        genesis.service_builder_share = service_builder_share;
    }

    if let Some(max_boost) = params.max_boost {
        genesis.max_boost = max_boost;
    }

    if let Some(supply_at_genesis) = params.supply_at_genesis {
        genesis.supply_at_genesis = supply_at_genesis;
    }
    let config = Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    };

    init_app(Some(config))
}

async fn simple_epoch_change(
    epoch: Epoch,
    committee_keystore: &Vec<GenesisCommitteeKeystore>,
    update_socket: &Socket<Block, BlockExecutionResponse>,
    query_runner: &QueryRunner,
) -> Result<()> {
    let required_signals = 2 * committee_keystore.len() / 3 + 1;
    // make call epoch change for 2/3rd committe members
    for (index, node) in committee_keystore.iter().enumerate().take(required_signals) {
        let nonce = query_runner
            .get_node_info(&node.node_secret_key.to_pk())
            .unwrap()
            .nonce
            + 1;
        let req = get_update_request_node(
            UpdateMethod::ChangeEpoch { epoch },
            &node.node_secret_key,
            nonce,
        );
        let res = run_transaction(vec![req.into()], update_socket).await?;
        // check epoch change
        if index == required_signals - 1 {
            assert!(res.change_epoch);
        }
    }
    Ok(())
}

// Helper method to get a transaction update request from a node.
// Passing the private key around like this should only be done for
// testing.
fn get_update_request_node(
    method: UpdateMethod,
    secret_key: &NodeSecretKey,
    nonce: u64,
) -> UpdateRequest {
    let payload = UpdatePayload { nonce, method };
    let digest = payload.to_digest();
    let signature = secret_key.sign(&digest);
    UpdateRequest {
        sender: secret_key.to_pk().into(),
        signature: signature.into(),
        payload,
    }
}
// Passing the private key around like this should only be done for
// testing.
fn get_update_request_account(
    method: UpdateMethod,
    secret_key: &AccountOwnerSecretKey,
    nonce: u64,
) -> UpdateRequest {
    let payload = UpdatePayload { nonce, method };
    let digest = payload.to_digest();
    let signature = secret_key.sign(&digest);
    UpdateRequest {
        sender: secret_key.to_pk().into(),
        signature: signature.into(),
        payload,
    }
}

fn get_genesis() -> (Genesis, Vec<GenesisNode>) {
    let genesis = Genesis::load().unwrap();

    (genesis.clone(), genesis.node_info)
}
// Helper methods for tests
// Passing the private key around like this should only be done for
// testing.
fn pod_request(
    secret_key: &NodeSecretKey,
    commodity: u128,
    service_id: u32,
    nonce: u64,
) -> UpdateRequest {
    get_update_request_node(
        UpdateMethod::SubmitDeliveryAcknowledgmentAggregation {
            commodity,  // units of data served
            service_id, // service 0 serving bandwidth
            proofs: vec![DeliveryAcknowledgment],
            metadata: None,
        },
        secret_key,
        nonce,
    )
}

async fn run_transaction(
    requests: Vec<TransactionRequest>,
    update_socket: &Socket<Block, BlockExecutionResponse>,
) -> Result<BlockExecutionResponse> {
    let res = update_socket
        .run(Block {
            transactions: requests,
            digest: [0; 32],
        })
        .await
        .map_err(|r| anyhow!(format!("{r:?}")))?;
    Ok(res)
}

async fn deposit(
    amount: HpUfixed<18>,
    token: Tokens,
    secret_key: &AccountOwnerSecretKey,
    update_socket: &Socket<Block, BlockExecutionResponse>,
    nonce: u64,
) {
    // Deposit some FLK into account 1
    let req = get_update_request_account(
        UpdateMethod::Deposit {
            proof: ProofOfConsensus {},
            token,
            amount,
        },
        secret_key,
        nonce,
    );
    run_transaction(vec![req.into()], update_socket)
        .await
        .unwrap();
}

async fn stake_lock(
    locked_for: u64,
    node: NodePublicKey,
    secret_key: &AccountOwnerSecretKey,
    update_socket: &Socket<Block, BlockExecutionResponse>,
    nonce: u64,
) {
    // Deposit some FLK into account 1
    let req = get_update_request_account(
        UpdateMethod::StakeLock { node, locked_for },
        secret_key,
        nonce,
    );
    run_transaction(vec![req.into()], update_socket)
        .await
        .unwrap();
}

async fn stake(
    amount: HpUfixed<18>,
    node_public_key: NodePublicKey,
    consensus_key: ConsensusPublicKey,
    secret_key: &AccountOwnerSecretKey,
    update_socket: &Socket<Block, BlockExecutionResponse>,
    nonce: u64,
) {
    let update = get_update_request_account(
        UpdateMethod::Stake {
            amount,
            node_public_key,
            consensus_key: Some(consensus_key),
            node_domain: Some("127.0.0.1".parse().unwrap()),
            worker_public_key: Some([0; 32].into()),
            worker_domain: Some("127.0.0.1".parse().unwrap()),
            ports: Some(NodePorts::default()),
        },
        secret_key,
        nonce,
    );
    if let TransactionResponse::Revert(error) = run_transaction(vec![update.into()], update_socket)
        .await
        .unwrap()
        .txn_receipts[0]
        .response
        .clone()
    {
        panic!("Stake reverted: {error:?}");
    }
}

#[test]
async fn test_genesis() {
    // Init application + get the query and update socket
    let (_, query_runner) = init_app(None);
    // Get the genesis paramaters plus the initial committee
    let (genesis, genesis_committee) = get_genesis();
    // For every member of the genesis committee they should have an initial stake of the min stake
    // Query to make sure that holds true
    for node in genesis_committee {
        let balance = query_runner.get_staked(&node.primary_public_key);
        assert_eq!(HpUfixed::<18>::from(genesis.min_stake), balance);
    }
}

#[test]
async fn test_epoch_change() {
    let (committee, keystore) = get_genesis_committee(4);
    let mut genesis = Genesis::load().unwrap();
    let committee_size = committee.len();
    genesis.node_info = committee;
    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));

    let required_signals = 2 * committee_size / 3 + 1;

    // Have (required_signals - 1) say they are ready to change epoch
    // make sure the epoch doesnt change each time someone signals
    for node in keystore.iter().take(required_signals - 1) {
        let req = get_update_request_node(
            UpdateMethod::ChangeEpoch { epoch: 0 },
            &node.node_secret_key,
            1,
        );

        let res = run_transaction(vec![req.into()], &update_socket)
            .await
            .unwrap();
        // Make sure epoch didnt change
        assert!(!res.change_epoch);
    }
    // check that the current epoch is still 0
    assert_eq!(query_runner.get_epoch_info().epoch, 0);

    // Have the last needed committee member signal the epoch change and make sure it changes
    let req = get_update_request_node(
        UpdateMethod::ChangeEpoch { epoch: 0 },
        &keystore[required_signals].node_secret_key,
        1,
    );
    let res = run_transaction(vec![req.into()], &update_socket)
        .await
        .unwrap();
    assert!(res.change_epoch);

    // Query epoch info and make sure it incremented to new epoch
    assert_eq!(query_runner.get_epoch_info().epoch, 1);
}

#[test]
async fn test_stake() {
    let (update_socket, query_runner) = init_app(None);
    let (genesis, _) = get_genesis();

    let owner_secret_key = AccountOwnerSecretKey::generate();
    let node_secret_key = NodeSecretKey::generate();

    // Deposit some FLK into account 1
    let update1 = get_update_request_account(
        UpdateMethod::Deposit {
            proof: ProofOfConsensus {},
            token: Tokens::FLK,
            amount: 1_000_u64.into(),
        },
        &owner_secret_key,
        1,
    );
    let update2 = get_update_request_account(
        UpdateMethod::Deposit {
            proof: ProofOfConsensus {},
            token: Tokens::FLK,
            amount: 1_000_u64.into(),
        },
        &owner_secret_key,
        2,
    );
    // Put 2 of the transaction in the block just to also test block exucution a bit
    run_transaction(vec![update1.into(), update2.into()], &update_socket)
        .await
        .unwrap();

    // check that he has 2_000 flk balance
    assert_eq!(
        query_runner.get_flk_balance(&owner_secret_key.to_pk().into()),
        2_000_u64.into()
    );

    // Test staking on a new node

    // First check that trying to stake without providing all the node info reverts
    let update = get_update_request_account(
        UpdateMethod::Stake {
            amount: 1_000_u64.into(),
            node_public_key: node_secret_key.to_pk(),
            consensus_key: None,
            node_domain: None,
            worker_public_key: None,
            worker_domain: None,
            ports: None,
        },
        &owner_secret_key,
        3,
    );
    let res = run_transaction(vec![update.into()], &update_socket)
        .await
        .unwrap();

    assert_eq!(
        TransactionResponse::Revert(ExecutionError::InsufficientNodeDetails),
        res.txn_receipts[0].response
    );

    // Now try with the correct details for a new node
    let update = get_update_request_account(
        UpdateMethod::Stake {
            amount: 1_000_u64.into(),
            node_public_key: node_secret_key.to_pk(),
            consensus_key: Some([0; 96].into()),
            node_domain: Some("127.0.0.1".parse().unwrap()),
            worker_public_key: Some([0; 32].into()),
            worker_domain: Some("127.0.0.1".parse().unwrap()),
            ports: Some(NodePorts::default()),
        },
        &owner_secret_key,
        4,
    );

    if let TransactionResponse::Revert(error) = run_transaction(vec![update.into()], &update_socket)
        .await
        .unwrap()
        .txn_receipts[0]
        .response
        .clone()
    {
        panic!("Stake reverted: {error:?}");
    }

    // Query the new node and make sure he has the proper stake
    assert_eq!(
        query_runner.get_staked(&node_secret_key.to_pk()),
        1_000_u64.into()
    );

    // Stake 1000 more but since it is not a new node we should be able to leave the optional
    // paramaters out without a revert
    let update = get_update_request_account(
        UpdateMethod::Stake {
            amount: 1_000_u64.into(),
            node_public_key: node_secret_key.to_pk(),
            consensus_key: None,
            node_domain: None,
            worker_public_key: None,
            worker_domain: None,
            ports: None,
        },
        &owner_secret_key,
        5,
    );
    if let TransactionResponse::Revert(error) = run_transaction(vec![update.into()], &update_socket)
        .await
        .unwrap()
        .txn_receipts[0]
        .response
        .clone()
    {
        panic!("Stake reverted: {error:?}");
    }

    // Node should now have 2_000 stake
    assert_eq!(
        query_runner.get_staked(&node_secret_key.to_pk()),
        2_000_u64.into()
    );

    // Now test unstake and make sure it moves the tokens to locked status
    let update = get_update_request_account(
        UpdateMethod::Unstake {
            amount: 1_000_u64.into(),
            node: node_secret_key.to_pk(),
        },
        &owner_secret_key,
        6,
    );
    run_transaction(vec![update.into()], &update_socket)
        .await
        .unwrap();

    // Check that his locked is 1000 and his remaining stake is 1000
    assert_eq!(
        query_runner.get_staked(&node_secret_key.to_pk()),
        1_000_u64.into()
    );
    assert_eq!(
        query_runner.get_locked(&node_secret_key.to_pk()),
        1_000_u64.into()
    );
    // Since this test starts at epoch 0 locked_until will be == lock_time
    assert_eq!(
        query_runner.get_locked_time(&node_secret_key.to_pk()),
        genesis.lock_time
    );

    // Try to withdraw the locked tokens and it should revery
    let update = get_update_request_account(
        UpdateMethod::WithdrawUnstaked {
            node: node_secret_key.to_pk(),
            recipient: None,
        },
        &owner_secret_key,
        7,
    );
    let res = run_transaction(vec![update.into()], &update_socket)
        .await
        .unwrap()
        .txn_receipts[0]
        .response
        .clone();
    assert_eq!(
        TransactionResponse::Revert(ExecutionError::TokensLocked),
        res
    );
}

#[test]
async fn test_stake_lock() {
    let (update_socket, query_runner) = init_app(None);

    let owner_secret_key = AccountOwnerSecretKey::generate();
    let node_secret_key = NodeSecretKey::generate();
    deposit(
        1_000_u64.into(),
        Tokens::FLK,
        &owner_secret_key,
        &update_socket,
        1,
    )
    .await;
    assert_eq!(
        query_runner.get_flk_balance(&owner_secret_key.to_pk().into()),
        1_000_u64.into()
    );

    stake(
        1_000_u64.into(),
        node_secret_key.to_pk(),
        [0; 96].into(),
        &owner_secret_key,
        &update_socket,
        2,
    )
    .await;
    assert_eq!(
        query_runner.get_staked(&node_secret_key.to_pk()),
        1_000_u64.into()
    );

    let stake_lock_req = get_update_request_account(
        UpdateMethod::StakeLock {
            node: node_secret_key.to_pk(),
            locked_for: 365,
        },
        &owner_secret_key,
        3,
    );

    if let TransactionResponse::Revert(error) =
        run_transaction(vec![stake_lock_req.into()], &update_socket)
            .await
            .unwrap()
            .txn_receipts[0]
            .response
            .clone()
    {
        panic!("Stake locking reverted: {error:?}");
    }
    assert_eq!(
        query_runner.get_stake_locked_until(&node_secret_key.to_pk()),
        365
    );

    let unstake_req = get_update_request_account(
        UpdateMethod::Unstake {
            amount: 1_000_u64.into(),
            node: node_secret_key.to_pk(),
        },
        &owner_secret_key,
        4,
    );
    let res = run_transaction(vec![unstake_req.into()], &update_socket)
        .await
        .unwrap()
        .txn_receipts[0]
        .response
        .clone();

    assert_eq!(
        res,
        TransactionResponse::Revert(ExecutionError::LockedTokensUnstakeForbidden)
    );
}

#[test]
async fn test_pod_without_proof() {
    let (committee, keystore) = get_genesis_committee(4);
    let mut genesis = Genesis::load().unwrap();
    genesis.node_info = committee;
    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));

    let bandwidth_pod = pod_request(&keystore[0].node_secret_key, 1000, 0, 1);
    let compute_pod = pod_request(&keystore[0].node_secret_key, 2000, 1, 2);

    // run the delivery ack transaction
    if let Err(e) = run_transaction(
        vec![bandwidth_pod.into(), compute_pod.into()],
        &update_socket,
    )
    .await
    {
        panic!("{e}");
    }

    assert_eq!(
        query_runner
            .get_node_served(&keystore[0].node_secret_key.to_pk())
            .served,
        vec![1000, 2000]
    );

    assert_eq!(
        query_runner.get_total_served(0),
        TotalServed {
            served: vec![1000, 2000],
            reward_pool: (0.1 * 1000_f64 + 0.2 * 2000_f64).into()
        }
    );
}

#[test]
async fn test_distribute_rewards() {
    let (committee, keystore) = get_genesis_committee(4);

    let max_inflation = 10;
    let protocol_part = 10;
    let node_part = 80;
    let service_part = 10;
    let boost = 4;
    let supply_at_genesis = 1_000_000;
    let (update_socket, query_runner) = init_app_with_params(
        Params {
            epoch_time: None,
            max_inflation: Some(max_inflation),
            protocol_share: Some(protocol_part),
            node_share: Some(node_part),
            service_builder_share: Some(service_part),
            max_boost: Some(boost),
            supply_at_genesis: Some(supply_at_genesis),
        },
        Some(committee),
    );

    // get params for emission calculations
    let percentage_divisor: HpUfixed<18> = 100_u16.into();
    let supply_at_year_start: HpUfixed<18> = supply_at_genesis.into();
    let inflation: HpUfixed<18> = HpUfixed::from(max_inflation) / &percentage_divisor;
    let node_share = HpUfixed::from(node_part) / &percentage_divisor;
    let protocol_share = HpUfixed::from(protocol_part) / &percentage_divisor;
    let service_share = HpUfixed::from(service_part) / &percentage_divisor;

    let owner_secret_key1 = AccountOwnerSecretKey::generate();
    let node_secret_key1 = NodeSecretKey::generate();
    let owner_secret_key2 = AccountOwnerSecretKey::generate();
    let node_secret_key2 = NodeSecretKey::generate();

    // deposit FLK tokens and stake it
    deposit(
        10_000_u64.into(),
        Tokens::FLK,
        &owner_secret_key1,
        &update_socket,
        1,
    )
    .await;
    stake(
        10_000_u64.into(),
        node_secret_key1.to_pk(),
        [0; 96].into(),
        &owner_secret_key1,
        &update_socket,
        2,
    )
    .await;
    deposit(
        10_000_u64.into(),
        Tokens::FLK,
        &owner_secret_key2,
        &update_socket,
        1,
    )
    .await;
    stake(
        10_000_u64.into(),
        node_secret_key2.to_pk(),
        [1; 96].into(),
        &owner_secret_key2,
        &update_socket,
        2,
    )
    .await;
    // staking locking for 4 year to get boosts
    stake_lock(
        1460,
        node_secret_key2.to_pk(),
        &owner_secret_key2,
        &update_socket,
        3,
    )
    .await;

    // submit pods for usage
    let pod_10 = pod_request(&node_secret_key1, 12_800, 0, 1);
    let pod11 = pod_request(&node_secret_key1, 3_600, 1, 2);
    let pod_21 = pod_request(&node_secret_key2, 5000, 1, 1);

    let node_1_usd = 0.1 * 12_800_f64 + 0.2 * 3_600_f64; // 2_000 in revenue
    let node_2_usd = 0.2 * 5_000_f64; // 1_000 in revenue
    let reward_pool: HpUfixed<6> = (node_1_usd + node_2_usd).into();

    let node_1_proportion: HpUfixed<18> = HpUfixed::from(2000_u64) / HpUfixed::from(3000_u64);
    let node_2_proportion: HpUfixed<18> = HpUfixed::from(1000_u64) / HpUfixed::from(3000_u64);

    let service_proportions: Vec<HpUfixed<18>> = vec![
        HpUfixed::from(1280_u64) / HpUfixed::from(3000_u64),
        HpUfixed::from(1720_u64) / HpUfixed::from(3000_u64),
    ];
    // run the delivery ack transaction
    if let Err(e) = run_transaction(
        vec![pod_10.into(), pod11.into(), pod_21.into()],
        &update_socket,
    )
    .await
    {
        panic!("{e}");
    }

    // call epoch change that will trigger distribute rewards
    if let Err(err) = simple_epoch_change(0, &keystore, &update_socket, &query_runner).await {
        panic!("error while changing epoch, {err}");
    }

    // assert stable balances
    let stables_balance = query_runner.get_stables_balance(&owner_secret_key1.to_pk().into());
    assert_eq!(
        stables_balance,
        <f64 as Into<HpUfixed<6>>>::into(node_1_usd) * node_share.convert_precision()
    );
    let stables_balance2 = query_runner.get_stables_balance(&owner_secret_key2.to_pk().into());
    assert_eq!(
        stables_balance2,
        <f64 as Into<HpUfixed<6>>>::into(node_2_usd) * node_share.convert_precision()
    );

    let total_share =
        &node_1_proportion * HpUfixed::from(1_u64) + &node_2_proportion * HpUfixed::from(4_u64);

    // calculate emissions per unit
    let emissions: HpUfixed<18> = (inflation * supply_at_year_start) / &365.0.into();
    let emissions_for_node = &emissions * &node_share;

    // assert flk balances node 1
    let node_flk_balance1 = query_runner.get_flk_balance(&owner_secret_key1.to_pk().into());
    let node_flk_rewards1 = (&emissions_for_node * &node_1_proportion) / &total_share;
    assert_eq!(node_flk_balance1, node_flk_rewards1);

    // assert flk balances node 2
    let node_flk_balance2 = query_runner.get_flk_balance(&owner_secret_key2.to_pk().into());
    let node_flk_rewards2: HpUfixed<18> =
        (&emissions_for_node * (&node_2_proportion * HpUfixed::from(4_u64))) / &total_share;
    assert_eq!(node_flk_balance2, node_flk_rewards2);

    // assert protocols share
    let protocol_account = query_runner.get_protocol_fund_address();
    let protocol_balance = query_runner.get_flk_balance(&protocol_account);
    let protocol_rewards = &emissions * &protocol_share;
    assert_eq!(protocol_balance, protocol_rewards);
    let protocol_stables_balance = query_runner.get_stables_balance(&protocol_account);
    assert_eq!(
        &reward_pool * &protocol_share.convert_precision(),
        protocol_stables_balance
    );

    // assert service balances with service id 0 and 1
    for s in 0..2 {
        let service_owner = query_runner.get_service_info(s).owner;
        let service_balance = query_runner.get_flk_balance(&service_owner);
        assert_eq!(
            service_balance,
            &emissions * &service_share * &service_proportions[s as usize]
        );
        let service_stables_balance = query_runner.get_stables_balance(&service_owner);
        assert_eq!(
            service_stables_balance,
            &reward_pool
                * &service_share.convert_precision()
                * &service_proportions[s as usize].convert_precision()
        );
    }
}

#[test]
async fn test_submit_rep_measurements() {
    let (committee, keystore) = get_genesis_committee(4);
    let mut genesis = Genesis::load().unwrap();
    genesis.node_info = committee;
    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));

    let mut map = BTreeMap::new();
    let mut rng = random::get_seedable_rng();

    let measurements1 = reputation::generate_reputation_measurements(&mut rng, 0.1);
    let peer1 = keystore[1].node_secret_key.to_pk();
    let peer_index1 = query_runner.pubkey_to_index(peer1).unwrap();
    map.insert(peer_index1, measurements1.clone());

    let measurements2 = reputation::generate_reputation_measurements(&mut rng, 0.1);
    let peer2 = keystore[2].node_secret_key.to_pk();
    let peer_index2 = query_runner.pubkey_to_index(peer2).unwrap();
    map.insert(peer_index2, measurements2.clone());

    let reporting_node_key = keystore[0].node_secret_key.to_pk();
    let reporting_node_index = query_runner.pubkey_to_index(reporting_node_key).unwrap();
    let req = get_update_request_node(
        UpdateMethod::SubmitReputationMeasurements { measurements: map },
        &keystore[0].node_secret_key,
        1,
    );
    if let Err(e) = run_transaction(vec![req.into()], &update_socket).await {
        panic!("{e}");
    }

    let rep_measurements1 = query_runner.get_rep_measurements(&peer_index1);
    assert_eq!(rep_measurements1.len(), 1);
    assert_eq!(rep_measurements1[0].reporting_node, reporting_node_index);
    assert_eq!(rep_measurements1[0].measurements, measurements1);

    let rep_measurements2 = query_runner.get_rep_measurements(&peer_index2);
    assert_eq!(rep_measurements2.len(), 1);
    assert_eq!(rep_measurements2[0].reporting_node, reporting_node_index);
    assert_eq!(rep_measurements2[0].measurements, measurements2);
}

#[test]
async fn test_rep_scores() {
    let (committee, keystore) = get_genesis_committee(4);
    let committee_len = committee.len();
    let mut genesis = Genesis::load().unwrap();
    genesis.node_info = committee;
    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));
    let required_signals = 2 * committee_len / 3 + 1;

    let mut rng = random::get_seedable_rng();

    let mut map = BTreeMap::new();
    let measurements = reputation::generate_reputation_measurements(&mut rng, 0.1);
    let peer1 = keystore[2].node_secret_key.to_pk();
    let peer_index1 = query_runner.pubkey_to_index(peer1).unwrap();
    map.insert(peer_index1, measurements.clone());

    let measurements = reputation::generate_reputation_measurements(&mut rng, 0.1);
    let peer2 = keystore[3].node_secret_key.to_pk();
    let peer_index2 = query_runner.pubkey_to_index(peer2).unwrap();
    map.insert(peer_index2, measurements.clone());

    let req = get_update_request_node(
        UpdateMethod::SubmitReputationMeasurements { measurements: map },
        &keystore[0].node_secret_key,
        1,
    );

    if let Err(e) = run_transaction(vec![req.into()], &update_socket).await {
        panic!("{e}");
    }

    let mut map = BTreeMap::new();
    let measurements = reputation::generate_reputation_measurements(&mut rng, 0.1);
    map.insert(peer_index1, measurements.clone());

    let measurements = reputation::generate_reputation_measurements(&mut rng, 0.1);
    map.insert(peer_index2, measurements.clone());

    let req = get_update_request_node(
        UpdateMethod::SubmitReputationMeasurements { measurements: map },
        &keystore[1].node_secret_key,
        1,
    );

    if let Err(e) = run_transaction(vec![req.into()], &update_socket).await {
        panic!("{e}");
    }

    // Change epoch so that rep scores will be calculated from the measurements.
    for (i, node) in keystore.iter().enumerate().take(required_signals) {
        // Not the prettiest solution but we have to keep track of the nonces somehow.
        let nonce = if i == 0 || i == 1 { 2 } else { 1 };
        let req = get_update_request_node(
            UpdateMethod::ChangeEpoch { epoch: 0 },
            &node.node_secret_key,
            nonce,
        );
        run_transaction(vec![req.into()], &update_socket)
            .await
            .unwrap();
    }

    assert!(query_runner.get_reputation(&peer_index1).is_some());
    assert!(query_runner.get_reputation(&peer_index2).is_some());
}

#[test]
async fn test_supply_across_epoch() {
    let (mut committee, mut keystore) = get_genesis_committee(4);

    let epoch_time = 100;
    let max_inflation = 10;
    let protocol_part = 10;
    let node_part = 80;
    let service_part = 10;
    let boost = 4;
    let supply_at_genesis = 1000000;
    let (update_socket, query_runner) = init_app_with_params(
        Params {
            epoch_time: Some(epoch_time),
            max_inflation: Some(max_inflation),
            protocol_share: Some(protocol_part),
            node_share: Some(node_part),
            service_builder_share: Some(service_part),
            max_boost: Some(boost),
            supply_at_genesis: Some(supply_at_genesis),
        },
        Some(committee.clone()),
    );

    // get params for emission calculations
    let percentage_divisor: HpUfixed<18> = 100_u16.into();
    let supply_at_year_start: HpUfixed<18> = supply_at_genesis.into();
    let inflation: HpUfixed<18> = HpUfixed::from(max_inflation) / &percentage_divisor;
    let node_share = HpUfixed::from(node_part) / &percentage_divisor;
    let protocol_share = HpUfixed::from(protocol_part) / &percentage_divisor;
    let service_share = HpUfixed::from(service_part) / &percentage_divisor;

    let owner_secret_key = AccountOwnerSecretKey::generate();
    let node_secret_key = NodeSecretKey::generate();
    let consensus_secret_key = ConsensusSecretKey::generate();

    // deposit FLK tokens and stake it
    deposit(
        10_000_u64.into(),
        Tokens::FLK,
        &owner_secret_key,
        &update_socket,
        1,
    )
    .await;
    stake(
        10_000_u64.into(),
        node_secret_key.to_pk(),
        consensus_secret_key.to_pk(),
        &owner_secret_key,
        &update_socket,
        2,
    )
    .await;
    // the index should be increment of whatever the size of genesis committee is, 5 in this case
    add_to_committee(
        &mut committee,
        &mut keystore,
        node_secret_key.clone(),
        consensus_secret_key.clone(),
        owner_secret_key.clone(),
        5,
    );
    // every epoch supply increase similar for simplicity of the test
    let _node_1_usd = 0.1 * 10000_f64;

    // calculate emissions per unit
    let emissions_per_epoch: HpUfixed<18> = (&inflation * &supply_at_year_start) / &365.0.into();

    let mut supply = supply_at_year_start;

    // 365 epoch changes to see if the current supply and year start suppply are ok
    for i in 0..365 {
        // add at least one transaction per epoch, so reward pool is not zero
        let nonce = query_runner
            .get_node_info(&node_secret_key.to_pk())
            .unwrap()
            .nonce;
        let pod_10 = pod_request(&node_secret_key, 10000, 0, nonce + 1);
        // run the delivery ack transaction
        if let TransactionResponse::Revert(error) =
            run_transaction(vec![pod_10.into()], &update_socket)
                .await
                .unwrap()
                .txn_receipts[0]
                .response
                .clone()
        {
            panic!("{error:?}");
        }
        let (_, new_keystore) = get_new_committee(&query_runner, &committee, &keystore);
        if let Err(err) = simple_epoch_change(i, &new_keystore, &update_socket, &query_runner).await
        {
            panic!("error while changing epoch, {err}");
        }

        let supply_increase = &emissions_per_epoch * &node_share
            + &emissions_per_epoch * &protocol_share
            + &emissions_per_epoch * &service_share;
        let total_supply = query_runner.get_total_supply();
        supply += supply_increase;
        assert_eq!(total_supply, supply);
        if i == 364 {
            // the supply_year_start should update
            let supply_year_start = query_runner.get_year_start_supply();
            assert_eq!(total_supply, supply_year_start);
        }
    }
}

#[test]
async fn test_validate_txn() {
    let (committee, keystore) = get_genesis_committee(4);
    let mut genesis = Genesis::load().unwrap();
    genesis.node_info = committee;
    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));

    // Submit a ChangeEpoch transaction that will revert (EpochHasNotStarted) and ensure that the
    // `validate_txn` method of the query runner returns the same response as the update runner.
    let req = get_update_request_node(
        UpdateMethod::ChangeEpoch { epoch: 1 },
        &keystore[0].node_secret_key,
        1,
    );
    let res = run_transaction(vec![req.clone().into()], &update_socket)
        .await
        .unwrap();
    let req = get_update_request_node(
        UpdateMethod::ChangeEpoch { epoch: 1 },
        &keystore[0].node_secret_key,
        2,
    );
    assert_eq!(
        res.txn_receipts[0].response,
        query_runner.validate_txn(req.into())
    );

    // Submit a ChangeEpoch transaction that will succeed and ensure that the
    // `validate_txn` method of the query runner returns the same response as the update runner.
    let req = get_update_request_node(
        UpdateMethod::ChangeEpoch { epoch: 0 },
        &keystore[0].node_secret_key,
        2,
    );
    let res = run_transaction(vec![req.into()], &update_socket)
        .await
        .unwrap();
    let req = get_update_request_node(
        UpdateMethod::ChangeEpoch { epoch: 0 },
        &keystore[1].node_secret_key,
        1,
    );
    assert_eq!(
        res.txn_receipts[0].response,
        query_runner.validate_txn(req.into())
    );
}

#[test]
async fn test_is_valid_node() {
    let (update_socket, query_runner) = init_app(None);

    let owner_secret_key = AccountOwnerSecretKey::generate();
    let node_secret_key = NodeSecretKey::generate();

    // Stake minimum required amount.
    let minimum_stake_amount = query_runner.get_staking_amount();
    deposit(
        minimum_stake_amount.into(),
        Tokens::FLK,
        &owner_secret_key,
        &update_socket,
        1,
    )
    .await;
    stake(
        minimum_stake_amount.into(),
        node_secret_key.to_pk(),
        [0; 96].into(),
        &owner_secret_key,
        &update_socket,
        2,
    )
    .await;
    // Make sure that this node is a valid node.
    assert!(query_runner.is_valid_node(&node_secret_key.to_pk()));

    // Generate new keys for a different node.
    let owner_secret_key = AccountOwnerSecretKey::generate();
    let node_secret_key = NodeSecretKey::generate();

    // Stake less than the minimum required amount.
    let less_than_minimum_skate_amount = minimum_stake_amount / 2;
    deposit(
        less_than_minimum_skate_amount.into(),
        Tokens::FLK,
        &owner_secret_key,
        &update_socket,
        1,
    )
    .await;
    stake(
        less_than_minimum_skate_amount.into(),
        node_secret_key.to_pk(),
        [1; 96].into(),
        &owner_secret_key,
        &update_socket,
        2,
    )
    .await;
    // Make sure that this node is not a valid node.
    assert!(!query_runner.is_valid_node(&node_secret_key.to_pk()));
}

#[test]
async fn test_get_node_registry() {
    let (committee, keystore) = get_genesis_committee(4);
    let mut genesis = Genesis::load().unwrap();
    genesis.node_info = committee;
    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));

    let owner_secret_key1 = AccountOwnerSecretKey::generate();
    let node_secret_key1 = NodeSecretKey::generate();

    // Stake minimum required amount.
    let minimum_stake_amount = query_runner.get_staking_amount();
    deposit(
        minimum_stake_amount.into(),
        Tokens::FLK,
        &owner_secret_key1,
        &update_socket,
        1,
    )
    .await;
    stake(
        minimum_stake_amount.into(),
        node_secret_key1.to_pk(),
        [0; 96].into(),
        &owner_secret_key1,
        &update_socket,
        2,
    )
    .await;

    // Generate new keys for a different node.
    let owner_secret_key2 = AccountOwnerSecretKey::generate();
    let node_secret_key2 = NodeSecretKey::generate();

    // Stake less than the minimum required amount.
    let less_than_minimum_skate_amount = minimum_stake_amount / 2;
    deposit(
        less_than_minimum_skate_amount.into(),
        Tokens::FLK,
        &owner_secret_key2,
        &update_socket,
        1,
    )
    .await;
    stake(
        less_than_minimum_skate_amount.into(),
        node_secret_key2.to_pk(),
        [1; 96].into(),
        &owner_secret_key2,
        &update_socket,
        2,
    )
    .await;

    // Generate new keys for a different node.
    let owner_secret_key3 = AccountOwnerSecretKey::generate();
    let node_secret_key3 = NodeSecretKey::generate();

    // Stake minimum required amount.
    deposit(
        minimum_stake_amount.into(),
        Tokens::FLK,
        &owner_secret_key3,
        &update_socket,
        1,
    )
    .await;
    stake(
        minimum_stake_amount.into(),
        node_secret_key3.to_pk(),
        [3; 96].into(),
        &owner_secret_key3,
        &update_socket,
        2,
    )
    .await;

    let valid_nodes = query_runner.get_node_registry(None);
    // We added two valid nodes, so the node registry should contain 2 nodes plus the committee.
    assert_eq!(valid_nodes.len(), 2 + keystore.len());
    let node_info1 = query_runner
        .get_node_info(&node_secret_key1.to_pk())
        .unwrap();
    // Node registry contains the first valid node
    assert!(valid_nodes.contains(&node_info1));
    let node_info2 = query_runner
        .get_node_info(&node_secret_key2.to_pk())
        .unwrap();
    // Node registry doesn't contain the invalid node
    assert!(!valid_nodes.contains(&node_info2));
    let node_info3 = query_runner
        .get_node_info(&node_secret_key3.to_pk())
        .unwrap();
    // Node registry contains the second valid node
    assert!(valid_nodes.contains(&node_info3));

    // Given some pagination parameters.
    let params = PagingParams {
        ignore_stake: true,
        start: 0,
        limit: keystore.len() + 3,
    };
    let nodes = query_runner.get_node_registry(Some(params));
    // We added 3 nodes, so the node registry should contain 3 nodes plus the committee.
    assert_eq!(nodes.len(), 3 + keystore.len());

    let params = PagingParams {
        ignore_stake: false,
        start: 0,
        limit: keystore.len() + 3,
    };
    let nodes = query_runner.get_node_registry(Some(params));
    // We added 2 valid nodes, so the node registry should contain 2 nodes plus the committee.
    assert_eq!(nodes.len(), 2 + keystore.len());

    let params = PagingParams {
        ignore_stake: true,
        start: 0,
        limit: keystore.len(),
    };
    let nodes = query_runner.get_node_registry(Some(params));
    // We get the first 4 nodes.
    assert_eq!(nodes.len(), keystore.len());

    let params = PagingParams {
        ignore_stake: true,
        start: 4,
        limit: keystore.len(),
    };
    let nodes = query_runner.get_node_registry(Some(params));
    // The first 4 nodes are the committee and we added 3 nodes.
    assert_eq!(nodes.len(), 3);

    let params = PagingParams {
        ignore_stake: false,
        start: keystore.len() as u32,
        limit: keystore.len(),
    };
    let valid_nodes = query_runner.get_node_registry(Some(params));
    // The first 4 nodes are the committee and we added 2 valid nodes.
    assert_eq!(valid_nodes.len(), 2);

    let params = PagingParams {
        ignore_stake: false,
        start: keystore.len() as u32,
        limit: 1,
    };
    let valid_nodes = query_runner.get_node_registry(Some(params));
    // The first 4 nodes are the committee and we added 3 nodes.
    assert_eq!(valid_nodes.len(), 1);
}

#[test]
async fn test_change_protocol_params() {
    let governance_secret_key = AccountOwnerSecretKey::generate();
    let governance_public_key = governance_secret_key.to_pk();

    let mut genesis = Genesis::load().unwrap();
    genesis.governance_address = governance_public_key.into();

    let (update_socket, query_runner) = init_app(Some(Config {
        genesis: Some(genesis),
        mode: Mode::Test,
        testnet: false,
        storage: StorageConfig::InMemory,
        db_path: None,
        db_options: None,
    }));

    let update_method = UpdateMethod::ChangeProtocolParam {
        param: ProtocolParams::LockTime,
        value: 5,
    };
    let update_request = get_update_request_account(update_method, &governance_secret_key, 1);
    run_transaction(vec![update_request.into()], &update_socket)
        .await
        .unwrap();
    assert_eq!(
        query_runner.get_protocol_params(ProtocolParams::LockTime),
        5
    );
    let update_method = UpdateMethod::ChangeProtocolParam {
        param: ProtocolParams::LockTime,
        value: 8,
    };
    let update_request = get_update_request_account(update_method, &governance_secret_key, 2);
    run_transaction(vec![update_request.into()], &update_socket)
        .await
        .unwrap();
    assert_eq!(
        query_runner.get_protocol_params(ProtocolParams::LockTime),
        8
    );
    // Make sure that another private key cannot change protocol parameters.
    let some_secret_key = AccountOwnerSecretKey::generate();

    let minimum_stake_amount = query_runner.get_staking_amount();
    deposit(
        minimum_stake_amount.into(),
        Tokens::FLK,
        &some_secret_key,
        &update_socket,
        1,
    )
    .await;

    let update_method = UpdateMethod::ChangeProtocolParam {
        param: ProtocolParams::LockTime,
        value: 1,
    };
    let update_request = get_update_request_account(update_method, &some_secret_key, 2);
    let response = run_transaction(vec![update_request.into()], &update_socket)
        .await
        .unwrap();
    assert_eq!(
        response.txn_receipts[0].response,
        TransactionResponse::Revert(ExecutionError::OnlyGovernance)
    );
    // Lock time should still be 8.
    assert_eq!(
        query_runner.get_protocol_params(ProtocolParams::LockTime),
        8
    );
}
