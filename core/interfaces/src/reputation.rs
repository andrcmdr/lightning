use std::time::Duration;

use infusion::c;
use lightning_types::NodeIndex;

use crate::config::ConfigConsumer;
use crate::infu_collection::Collection;
use crate::notifier::NotifierInterface;
use crate::signer::SubmitTxSocket;
use crate::{ApplicationInterface, ConfigProviderInterface, SignerInterface, WithStartAndShutdown};

#[infusion::service]
pub trait ReputationAggregatorInterface<C: Collection>:
    ConfigConsumer + Sized + WithStartAndShutdown
{
    fn _init(
        config: ::ConfigProviderInterface,
        signer: ::SignerInterface,
        notifier: ::NotifierInterface,
        app: ::ApplicationInterface,
    ) {
        Self::init(
            config.get::<Self>(),
            signer.get_socket(),
            notifier.clone(),
            app.sync_query(file!(), line!()),
        )
    }

    /// The reputation reporter can be used by our system to report the reputation of other
    type ReputationReporter: ReputationReporterInterface;

    /// The query runner can be used to query the local reputation of other nodes.
    type ReputationQuery: ReputationQueryInteface;

    /// Create a new reputation
    fn init(
        config: Self::Config,
        submit_tx: SubmitTxSocket,
        notifier: c!(C::NotifierInterface),
        query_runner: c!(C::ApplicationInterface::SyncExecutor),
    ) -> anyhow::Result<Self>;

    /// Returns a reputation reporter that can be used to capture interactions that we have
    /// with another peer.
    fn get_reporter(&self) -> Self::ReputationReporter;

    /// Returns a reputation query that can be used to answer queries about the local
    /// reputation we have of another peer.
    fn get_query(&self) -> Self::ReputationQuery;
}

/// Used to answer queries about the (local) reputation of other nodes, this queries should
/// be as real-time as possible, meaning that the most recent data captured by the reporter
/// should be taken into account at this layer.
#[infusion::blank]
pub trait ReputationQueryInteface: Clone + Send + Sync {
    /// Returns the reputation of the provided node locally.
    fn get_reputation_of(&self, peer: &NodeIndex) -> Option<u8>;
}

/// Reputation reporter is a cheaply cleanable object which can be used to report the interactions
/// that we have with another peer, this interface allows a reputation aggregator to spawn many
/// reporters which can use any method to report the data they capture to their aggregator so
/// that it can send it to the application layer.
#[infusion::blank]
pub trait ReputationReporterInterface: Clone + Send + Sync {
    /// Report a satisfactory (happy) interaction with the given peer. Used for up time.
    fn report_sat(&self, peer: NodeIndex, weight: Weight);

    /// Report a unsatisfactory (happy) interaction with the given peer. Used for down time.
    fn report_unsat(&self, peer: NodeIndex, weight: Weight);

    /// Report a ping interaction with another peer and the latency if the peer responded.
    /// `None` indicates that the peer did not respond.
    fn report_ping(&self, peer: NodeIndex, latency: Option<Duration>);

    /// Report the number of (healthy) bytes which we received from another peer.
    fn report_bytes_received(&self, peer: NodeIndex, bytes: u64, duration: Option<Duration>);

    /// Report the number of (healthy) bytes which we sent from another peer.
    fn report_bytes_sent(&self, peer: NodeIndex, bytes: u64, duration: Option<Duration>);

    /// Report the number of hops we have witnessed to the given peer.
    fn report_hops(&self, peer: NodeIndex, hops: u8);
}

// TODO: Move to types/reputation.rs as `ReputationWeight`.
#[derive(Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub enum Weight {
    Weak,
    Strong,
    VeryStrong,
    Provable,
}
