use cid::multihash::{Code, MultihashDigest};
use cid::Cid;
use lightning_blockstore::blockstore::Blockstore;
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::{
    partial,
    BlockStoreInterface,
    OriginProviderInterface,
    WithStartAndShutdown,
};
use lightning_test_utils::ipfs_gateway::spawn_gateway;

use crate::config::{Config, Gateway, Protocol};
use crate::IPFSOrigin;

partial!(TestBinding {
    OriginProviderInterface = IPFSOrigin<Self>;
    BlockStoreInterface = Blockstore<Self>;
});

#[tokio::test]
async fn test_origin() {
    let req_cid =
        Cid::try_from("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi").unwrap();
    let mut config = Config::default();

    let blockstore = Blockstore::<TestBinding>::init(Default::default()).unwrap();

    let req_fut = async move {
        config.gateways.push(Gateway {
            protocol: Protocol::Http,
            authority: "127.0.0.1:30100".to_string(),
        });
        let ipfs_origin = IPFSOrigin::<TestBinding>::init(config, blockstore.clone()).unwrap();
        ipfs_origin.start().await;

        let socket = ipfs_origin.get_socket();
        let hash = socket.run(req_cid.to_bytes()).await.unwrap().unwrap();

        let bytes = blockstore.read_all_to_vec(&hash).await.unwrap();
        assert!(
            Code::try_from(req_cid.hash().code())
                .ok()
                .map(|code| &code.digest(&bytes) == req_cid.hash())
                .unwrap()
        );
    };

    tokio::select! {
        _ = spawn_gateway(30100) => {

        }
        _ = req_fut => {

        }

    }
}

#[tokio::test]
async fn test_shutdown() {
    let blockstore = Blockstore::<TestBinding>::init(Default::default()).unwrap();
    let ipfs_origin = IPFSOrigin::<TestBinding>::init(Config::default(), blockstore).unwrap();
    assert!(!ipfs_origin.is_running());
    ipfs_origin.start().await;
    assert!(ipfs_origin.is_running());
    ipfs_origin.shutdown().await;
    assert!(!ipfs_origin.is_running());
}
