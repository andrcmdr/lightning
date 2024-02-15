use std::sync::Arc;

use fleek_crypto::NodePublicKey;
use infusion::c;

use crate::infu_collection::Collection;
use crate::{ApplicationInterface, ConfigConsumer, ConfigProviderInterface, SignerInterface};

/// The algorithm used for clustering our network and dynamically creating a network topology.
/// This clustering is later used in other parts of the codebase when connection to other nodes
/// is required. The gossip layer is an example of a component that can feed the data this
/// algorithm generates.
#[infusion::service]
pub trait TopologyInterface<C: Collection>: ConfigConsumer + Sized + Send + Sync + Clone {
    fn _init(
        config: ::ConfigProviderInterface,
        signer: ::SignerInterface,
        app: ::ApplicationInterface,
    ) {
        Self::init(
            config.get::<Self>(),
            signer.get_ed25519_pk(),
            app.sync_query(file!(), line!()),
        )
    }

    /// Create an instance of the structure from the provided configuration and public key.
    /// The public key is supposed to be the public key of our own node. This can be obtained
    /// from a [Signer](crate::SignerInterface). But making the `TopologyInterface` depend on
    /// SignerInterface seems odd and we want to avoid that.
    ///
    /// Due to that reason instead we just pass the public key here. For consistency and
    /// correctness of the implementation it is required that this public key to be our
    /// actual public key which is obtained from [get_bls_pk](crate::SignerInterface::get_bls_pk).
    fn init(
        config: Self::Config,
        our_public_key: NodePublicKey,
        query_runner: c!(C::ApplicationInterface::SyncExecutor),
    ) -> anyhow::Result<Self>;

    /// Suggest a list of connections that our current node must connect to. This should be
    /// according to the `our_public_key` value passed during the initialization.
    ///
    /// The result of this call is a 2-dimensional array, the first dimension determines the
    /// closeness of the nodes, the further items are the outer layer of the connections.
    ///
    /// This should return the result for the latest epoch. The [`TopologyInterface`] is poll
    /// based and implementations are recommended to cache the result of this computation.
    fn suggest_connections(&self) -> Arc<Vec<Vec<NodePublicKey>>>;
}
