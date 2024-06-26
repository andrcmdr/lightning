use std::fs;
use std::time::SystemTime;

use anyhow::Result;
use clap::Parser;
use fleek_crypto::PublicKey;
use lightning_application::app::Application;
use lightning_e2e::swarm::Swarm;
use lightning_e2e::utils::shutdown;
use lightning_interfaces::prelude::*;
use lightning_test_utils::logging;
use lightning_topology::Topology;
use resolved_pathbuf::ResolvedPathBuf;

partial!(PartialBinding {
    ApplicationInterface = Application<Self>;
    TopologyInterface = Topology<Self>;
});

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of nodes to spawn
    #[arg(short, long, default_value_t = 4)]
    num_nodes: usize,

    /// Number of committee members
    #[arg(short, long, default_value_t = 4)]
    committee_size: usize,

    /// Epoch duration in millis
    #[arg(short, long, default_value_t = 60000)]
    epoch_time: u64,

    /// Use persistence for the application state
    #[arg(short, long, default_value_t = false)]
    persistence: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::setup();

    let args = Cli::parse();

    if args.committee_size > args.num_nodes {
        panic!("Committee size can not be larger than number of nodes.")
    }

    let epoch_start = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let path = ResolvedPathBuf::try_from("~/.lightning-test/e2e/spawn-swarm").unwrap();
    if path.exists() {
        fs::remove_dir_all(&path).expect("Failed to clean up swarm directory before test.");
    }
    let swarm = Swarm::builder()
        .with_directory(path)
        .with_min_port(12000)
        .with_num_nodes(args.num_nodes)
        .with_committee_size(args.committee_size as u64)
        .with_epoch_time(args.epoch_time)
        .with_epoch_start(epoch_start)
        .with_archiver()
        .persistence(args.persistence)
        .build();
    swarm.launch().await.unwrap();

    let mut s = String::from("#####################################\n\n");
    for (pub_key, rpc_address) in swarm.get_rpc_addresses() {
        s.push_str(&format!(
            "Public Key: {}\nRPC Address: {}\n\n",
            pub_key.to_base58(),
            rpc_address
        ));
    }
    s.push_str("#####################################");
    println!("{s}");

    shutdown::shutdown_stream().await;

    Ok(())
}
