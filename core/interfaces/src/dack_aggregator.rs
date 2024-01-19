use affair::Socket;

use crate::common::WithStartAndShutdown;
use crate::config::ConfigConsumer;
use crate::infu_collection::Collection;
use crate::signer::SubmitTxSocket;
use crate::types::DeliveryAcknowledgment;
use crate::{ConfigProviderInterface, SignerInterface};

/// The socket which upon receiving a delivery acknowledgment can add it to the aggregator
/// queue which will later roll up a batch of delivery acknowledgments to the consensus.
pub type DeliveryAcknowledgmentSocket = Socket<DeliveryAcknowledgment, ()>;

#[infusion::service]
pub trait DeliveryAcknowledgmentAggregatorInterface<C: Collection>:
    WithStartAndShutdown + ConfigConsumer + Sized + Send + Sync
{
    fn _init(config: ::ConfigProviderInterface, signer: ::SignerInterface) {
        Self::init(config.get::<Self>(), signer.get_socket())
    }

    /// Initialize a new delivery acknowledgment aggregator.
    fn init(config: Self::Config, submit_tx: SubmitTxSocket) -> anyhow::Result<Self>;

    /// Returns the socket that can be used to submit delivery acknowledgments to be aggregated.
    fn socket(&self) -> DeliveryAcknowledgmentSocket;
}

pub trait LaneManager {}
