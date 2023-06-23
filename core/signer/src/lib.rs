mod config;
use async_trait::async_trait;
use config::Config;
use draco_interfaces::{
    common::WithStartAndShutdown,
    config::ConfigConsumer,
    signer::{SignerInterface, SubmitTxSocket},
    MempoolSocket,
};
use fleek_crypto::{
    NodeNetworkingPublicKey, NodeNetworkingSecretKey, NodePublicKey, NodeSecretKey, NodeSignature,
};

#[derive(Clone)]
pub struct Signer {}

#[async_trait]
impl WithStartAndShutdown for Signer {
    /// Returns true if this system is running or not.
    fn is_running(&self) -> bool {
        todo!()
    }

    /// Start the system, should not do anything if the system is already
    /// started.
    async fn start(&self) {
        todo!()
    }

    /// Send the shutdown signal to the system.
    async fn shutdown(&self) {
        todo!()
    }
}

#[async_trait]
impl SignerInterface for Signer {
    /// Initialize the signature service.
    async fn init(_config: Self::Config) -> anyhow::Result<Self> {
        todo!()
    }

    /// Provide the signer service with the mempool socket after initialization, this function
    /// should only be called once.
    fn provide_mempool(&mut self, _mempool: MempoolSocket) {
        todo!()
    }

    /// Returns the `BLS` public key of the current node.
    fn get_bls_pk(&self) -> NodePublicKey {
        todo!()
    }

    /// Returns the `Ed25519` (network) public key of the current node.
    fn get_ed25519_pk(&self) -> NodeNetworkingPublicKey {
        todo!()
    }

    /// Returns the loaded secret key material.
    ///
    /// # Safety
    ///
    /// Just like any other function which deals with secret material this function should
    /// be used with the greatest caution.
    fn get_sk(&self) -> (NodeNetworkingSecretKey, NodeSecretKey) {
        todo!()
    }

    /// Returns a socket that can be used to submit transactions to the mempool, these
    /// transactions are signed by the node and a proper nonce is assigned by the
    /// implementation.
    ///
    /// # Panics
    ///
    /// This function can panic if there has not been a prior call to `provide_mempool`.
    fn get_socket(&self) -> SubmitTxSocket {
        todo!()
    }

    /// Sign the provided raw digest and return a signature.
    ///
    /// # Safety
    ///
    /// This function is unsafe to use without proper reasoning, which is trivial since
    /// this function is responsible for signing arbitrary messages from other parts of
    /// the system.
    fn sign_raw_digest(&self, _digest: &[u8; 32]) -> NodeSignature {
        todo!()
    }
}

impl ConfigConsumer for Signer {
    const KEY: &'static str = "signer";

    type Config = Config;
}
