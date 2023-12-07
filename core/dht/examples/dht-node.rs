use std::net::SocketAddr;
use std::time::Duration;

use clap::{Parser, Subcommand};
use fleek_crypto::{NodePublicKey, NodeSecretKey, PublicKey, SecretKey};
use lightning_application::app::Application;
use lightning_application::config::{Mode, StorageConfig};
use lightning_dht::config::Config;
use lightning_dht::dht::Dht;
use lightning_interfaces::dht::{DhtInterface, DhtSocket};
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::{Blake3Hash, DhtRequest, DhtResponse, KeyPrefix};
use lightning_interfaces::{partial, ApplicationInterface, WithStartAndShutdown};
use lightning_topology::Topology;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, group = "bootstrap_address")]
    bootstrapper: Option<String>,

    #[arg(long, group = "bootstrap_key")]
    bootstrapper_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[group(required = true)]
enum Commands {
    Get {
        #[arg(short, long)]
        key: String,
    },
    Put,
    Join,
    Bootstrapper,
}

partial!(PartialBinding {
    ApplicationInterface = Application<Self>;
    TopologyInterface = Topology<Self>;
    DhtInterface = Dht<Self>;
});

#[tokio::main]
async fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let bootstrap_key_pem = include_str!("../../test-utils/keys/test_node.pem");
    let bootstrap_secret_key = NodeSecretKey::decode_pem(bootstrap_key_pem).unwrap();

    let bootstrap_key = match cli.bootstrapper_key {
        Some(bootstrapper_key) => NodePublicKey::from_base58(&bootstrapper_key)
            .expect("Failed to parse bootstrap public key"),
        None => bootstrap_secret_key.to_pk(),
    };

    match cli.command {
        Commands::Get { key } => {
            let address: SocketAddr = cli.bootstrapper.unwrap().parse().unwrap();
            let secret_key = NodeSecretKey::generate();
            tracing::info!("public key: {:?}", secret_key.to_pk());
            let dht_socket = start_node(secret_key, Some((address, bootstrap_key))).await;

            tracing::info!("GET {key:?}");

            let key = hex::decode(key).unwrap();

            let value = dht_socket
                .run(DhtRequest::Get {
                    prefix: KeyPrefix::ContentRegistry,
                    key,
                })
                .await
                .expect("sending get request failed.");

            if let DhtResponse::Get(Some(value)) = value {
                tracing::info!("value found is {:?}", value.value);
            }
        },
        Commands::Put => {
            let address: SocketAddr = cli.bootstrapper.unwrap().parse().unwrap();
            let secret_key = NodeSecretKey::generate();
            tracing::info!("public key: {:?}", secret_key.to_pk());
            let dht_socket = start_node(secret_key, Some((address, bootstrap_key))).await;

            // Todo: get actual hash.
            let key: Blake3Hash = rand::random();
            let value: [u8; 4] = rand::random();

            tracing::info!("PUT {}:{value:?}", hex::encode(key));

            dht_socket
                .run(DhtRequest::Put {
                    prefix: KeyPrefix::ContentRegistry,
                    key: key.to_vec(),
                    value: value.to_vec(),
                })
                .await
                .expect("sending put request failed.");

            // Todo: Let's remove this loop.
            // We have this loop so that the spawn task of `put` finishes.
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        },
        Commands::Join => {
            let address: SocketAddr = cli.bootstrapper.unwrap().parse().unwrap();
            let secret_key = NodeSecretKey::generate();
            tracing::info!("public key: {:?}", secret_key.to_pk());
            let _ = start_node(secret_key, Some((address, bootstrap_key))).await;
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        },
        Commands::Bootstrapper => {
            let _socket = start_node(bootstrap_secret_key, None).await;
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        },
    }

    tracing::info!("shutting down dht-node");
}

async fn start_node(
    secret_key: NodeSecretKey,
    bootstrapper: Option<(SocketAddr, NodePublicKey)>,
) -> DhtSocket {
    let application = Application::<PartialBinding>::init(
        lightning_application::config::Config {
            mode: Mode::Test,
            genesis: None,
            testnet: false,
            storage: StorageConfig::InMemory,
            db_path: None,
            db_options: None,
        },
        Default::default(),
    )
    .unwrap();
    let _query_runner = application.sync_query();
    let mut builder = Builder::<PartialBinding>::new(
        secret_key,
        Config::default(),
        Default::default(),
        Default::default(),
    );

    if let Some((address, key)) = bootstrapper {
        tracing::info!("bootstrapping to {address:?} {key:?}");
        builder.add_node(key, address);
    }

    let dht = builder.build().unwrap();
    let socket = dht.get_socket();
    dht.start().await;

    socket
}
