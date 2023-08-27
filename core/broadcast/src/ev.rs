//! This file contains the implementation of the main event loop of the broadcast. You
//! can think of broadcast as a single-threaded event loop. This allows the broadcast
//! to hold mutable references to its internal state and avoid lock.
//!
//! Given most events being handled here are mutating a shared object. We can avoid any
//! implementation complexity or performance drops due to use of locks by just handling
//! the entire thing in a central event loop.

use std::sync::Arc;

use fleek_crypto::{NodePublicKey, NodeSecretKey, NodeSignature, PublicKey, SecretKey};
use infusion::c;
use ink_quill::ToDigest;
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::{NodeIndex, Topic};
use lightning_interfaces::{
    ApplicationInterface,
    ConnectionPoolInterface,
    ConnectorInterface,
    ListenerConnector,
    ListenerInterface,
    NotifierInterface,
    PoolReceiver,
    PoolSender,
    SenderInterface,
    SenderReceiver,
    SyncQueryRunnerInterface,
    TopologyInterface,
};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::command::{Command, CommandReceiver, CommandSender, RecvCmd, SendCmd, SharedMessage};
use crate::db::Database;
use crate::frame::Frame;
use crate::interner::Interner;
use crate::peers::{ConnectionOrigin, ConnectionStatus, Peers};
use crate::recv_buffer::RecvBuffer;
use crate::ring::MessageRing;
use crate::stats::{ConnectionStats, Stats};
use crate::{Advr, Digest, Message, Want};

// TODO(qti3e): Move this to somewhere else.
pub type Topology = Arc<Vec<Vec<NodePublicKey>>>;

// The connection pool sender and receiver types.
type S<C> = PoolSender<C, c![C::ConnectionPoolInterface], Frame>;
type R<C> = PoolReceiver<C, c![C::ConnectionPoolInterface], Frame>;

/// The execution context of the broadcast.
pub struct Context<C: Collection> {
    /// Our database where we store what we have seen.
    db: Database,
    /// Our digest interner.
    interner: Interner,
    /// Managers of incoming message queue for each topic.
    incoming_messages: [RecvBuffer; 3],
    /// The state related to the connected peers that we have right now.
    peers: Peers<S<C>, R<C>>,
    /// The instance of stats collector.
    stats: Stats,
    /// We use this socket to let the main event loop know that we have established
    /// a connection with another node.
    new_outgoing_connection_tx: mpsc::UnboundedSender<(S<C>, R<C>)>,
    /// The receiver end of the above socket.
    new_outgoing_connection_rx: mpsc::UnboundedReceiver<(S<C>, R<C>)>,
    /// The channel which sends the commands to the event loop.
    command_tx: CommandSender,
    /// Receiving end of the commands.
    command_rx: CommandReceiver,
    sqr: c![C::ApplicationInterface::SyncExecutor],
    notifier: c![C::NotifierInterface],
    topology: c![C::TopologyInterface],
    listener: c![C::ConnectionPoolInterface::Listener<Frame>],
    connector: c![C::ConnectionPoolInterface::Connector<Frame>],
    sk: NodeSecretKey,
    pk: NodePublicKey,
    current_node_index: NodeIndex,
}

impl<C: Collection> Context<C> {
    pub fn new(
        db: Database,
        sqr: c![C::ApplicationInterface::SyncExecutor],
        notifier: c![C::NotifierInterface],
        topology: c![C::TopologyInterface],
        listener: c![C::ConnectionPoolInterface::Listener<Frame>],
        connector: c![C::ConnectionPoolInterface::Connector<Frame>],
        sk: NodeSecretKey,
    ) -> Self {
        let (new_outgoing_connection_tx, new_outgoing_connection_rx) = mpsc::unbounded_channel();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let pk = sk.to_pk();
        let peers = Peers::default();
        let stats = peers.stats.clone();
        Self {
            db,
            interner: Interner::new(u16::MAX),
            incoming_messages: [
                MessageRing::new(2048).into(),
                MessageRing::new(512).into(),
                MessageRing::new(1).into(),
            ],
            peers,
            stats,
            new_outgoing_connection_tx,
            new_outgoing_connection_rx,
            command_tx,
            command_rx,
            sqr,
            notifier,
            topology,
            listener,
            connector,
            sk,
            pk,
            current_node_index: 0, // will be set upon spawn.
        }
    }

    pub fn command_sender(&self) -> CommandSender {
        self.command_tx.clone()
    }

    pub fn spawn(self) -> (oneshot::Sender<()>, JoinHandle<Self>) {
        let (shutdown_tx, shutdown) = oneshot::channel();
        let handle = tokio::spawn(async move { main_loop(shutdown, self).await });
        (shutdown_tx, handle)
    }

    /// Try to lookup a node index from a private key in an efficient way by first looking up the
    /// connected peers mappings and then resorting to the application's query runner.
    fn get_node_index(&self, pk: &NodePublicKey) -> Option<NodeIndex> {
        self.peers
            .get_node_index(pk)
            .or_else(|| self.sqr.pubkey_to_index(*pk))
    }

    fn get_node_pk(&self, index: NodeIndex) -> Option<NodePublicKey> {
        self.sqr.index_to_pubkey(index)
    }

    fn apply_topology(&mut self, new_topology: Topology) {
        self.peers.unpin_all();

        for pk in new_topology.iter().flatten().copied() {
            self.peers.pin_peer(pk);

            if self.peers.get_connection_status(&pk) != ConnectionStatus::Closed {
                // TODO(qti3e): Should we maybe handle == ConnectionStatus::Open
                // instead.
                continue;
            }

            let tx = self.new_outgoing_connection_tx.clone();
            let connector = self.connector.clone();
            tokio::spawn(async move {
                let Some((sender, receiver)) = connector.connect(&pk).await else {
                    return;
                };

                tx.send((sender, receiver));
            });
        }

        self.peers.disconnect_unpinned();
    }

    /// Handle a message sent from a user.
    fn handle_frame(&mut self, sender: NodePublicKey, frame: Frame) {
        match frame {
            Frame::Advr(advr) => {
                self.handle_advr(sender, advr);
            },
            Frame::Want(want) => {
                self.handle_want(sender, want);
            },
            Frame::Message(msg) => {
                self.handle_message(sender, msg);
            },
        }
    }

    fn handle_advr(&mut self, sender: NodePublicKey, advr: Advr) {}

    fn handle_want(&mut self, sender: NodePublicKey, req: Want) {
        let id = req.interned_id;
        let Some(digest) = self.interner.get(id) else { return; };
        let Some(message) = self.db.get_message(digest) else { return; };
        self.peers.send_message(&sender, Frame::Message(message));
    }

    fn handle_message(&mut self, sender: NodePublicKey, msg: Message) {
        let Some(origin_pk) = self.get_node_pk(msg.origin) else {
            let index = self.get_node_index(&sender).unwrap();
            self.stats.report(index, ConnectionStats {
                invalid_messages_received_from_peer: 1,
                ..Default::default()
            });
            return;
        };

        let digest = msg.to_digest();
        if !origin_pk.verify(&msg.signature, &digest) {
            let index = self.get_node_index(&sender).unwrap();
            self.stats.report(
                index,
                ConnectionStats {
                    invalid_messages_received_from_peer: 1,
                    ..Default::default()
                },
            );
            return;
        }

        let topic_index = topic_to_index(msg.topic);

        let shared = SharedMessage {
            digest,
            origin: msg.origin,
            payload: msg.payload.into(),
        };

        self.incoming_messages[topic_index].insert(shared);
    }

    /// Handle a command sent from the mainland. Can be the broadcast object or
    /// a pubsub object.
    fn handle_command(&mut self, command: Command) {
        match command {
            Command::Recv(cmd) => self.handle_recv_cmd(cmd),
            Command::Send(cmd) => self.handle_send_cmd(cmd),
            Command::Propagate(digest) => self.handle_propagate_cmd(digest),
            Command::MarkInvalidSender(digest) => self.handle_mark_invalid_sender_cmd(digest),
        }
    }

    fn handle_recv_cmd(&mut self, cmd: RecvCmd) {
        let index = topic_to_index(cmd.topic);
        self.incoming_messages[index].response_to(cmd.last_seen, cmd.response);
    }

    fn handle_send_cmd(&mut self, cmd: SendCmd) {
        let (digest, message) = {
            let mut tmp = Message {
                origin: self.current_node_index,
                signature: NodeSignature([0; 64]),
                topic: cmd.topic,
                payload: cmd.payload,
            };
            let digest = tmp.to_digest();
            tmp.signature = self.sk.sign(&digest);
            (digest, tmp)
        };

        let id = self.interner.insert(digest);
        self.db.insert_with_message(id, digest, message);

        // Start advertising the message.
        self.peers.advertise(id, digest);
    }

    fn handle_propagate_cmd(&mut self, digest: Digest) {
        todo!()
    }

    fn handle_mark_invalid_sender_cmd(&mut self, digest: Digest) {
        todo!()
    }
}

/// Runs the main loop of the broadcast algorithm. This is our main central worker.
async fn main_loop<C: Collection>(
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
    mut ctx: Context<C>,
) -> Context<C> {
    // Subscribe to the changes from the topology.
    let mut topology_subscriber =
        spawn_topology_subscriber::<C>(ctx.notifier.clone(), ctx.topology.clone());

    // Provide the peers list with the index of our current node. It will need it
    // for resolving connection ordering disputes.
    // This could have been done during initialization, except at that point we don't
    // know if application has started and therefore if the database is loaded yet.
    ctx.current_node_index = ctx
        .sqr
        .pubkey_to_index(ctx.pk)
        .expect("Current node on the application state.");
    ctx.peers.set_current_node_index(ctx.current_node_index);

    loop {
        tokio::select! {
            // We kind of care about the priority of these events. Or do we?
            // TODO(qti3e): Evaluate this.
            // biased;

            // Prioritize the shutdown signal over everything.
            _ = &mut shutdown => {
                break;
            },

            // A command has been sent from the mainland. Process it.
            Some(command) = ctx.command_rx.recv() => {
                ctx.handle_command(command);
            },

            // The `topology_subscriber` recv can potentially return `None` when the
            // application execution socket is dropped. At that point we're also shutting
            // down anyway.
            Some(new_topology) = topology_subscriber.recv() => {
                ctx.apply_topology(new_topology);
            },

            Some(conn) = ctx.new_outgoing_connection_rx.recv() => {
                let Some(index) = ctx.get_node_index(conn.0.pk()) else {
                    continue;
                };
                ctx.peers.handle_new_connection(ConnectionOrigin::Us , index, conn);
            },

            Some((sender, frame)) = ctx.peers.recv() => {
                ctx.handle_frame(sender, frame);
            },

            // Handle the case when another node is dialing us.
            // TODO(qti3e): Is this cancel safe?
            Some(conn) = ctx.listener.accept() => {
                let Some(index) = ctx.get_node_index(conn.0.pk()) else {
                    continue;
                };
                ctx.peers.handle_new_connection(ConnectionOrigin::Remote , index, conn);
            },
        }
    }

    // Handover over the context.
    ctx
}

/// Spawn a task that listens for epoch changes and sends the new topology through an `mpsc`
/// channel.
///
/// This function will always compute/fire the current topology as the first message before waiting
/// for the next epoch.
///
/// This function does not block the caller and immediately returns.
fn spawn_topology_subscriber<C: Collection>(
    notifier: c![C::NotifierInterface],
    topology: c![C::TopologyInterface],
) -> mpsc::Receiver<Topology> {
    // Create the output channel from which we send out the computed
    // topology.
    let (w_tx, w_rx) = mpsc::channel(64);

    // Subscribe to new epochs coming in.
    let (tx, mut rx) = mpsc::channel(64);
    notifier.notify_on_new_epoch(tx);

    tokio::spawn(async move {
        'spell: loop {
            let topology = topology.clone();

            // Computing the topology might be a blocking task.
            let result = tokio::task::spawn_blocking(move || topology.suggest_connections());

            let topology = result.await.expect("Failed to compute topology.");

            if w_tx.send(topology).await.is_err() {
                // Nobody is interested in hearing what we have to say anymore.
                // This can happen when all instance of receivers are dropped.
                break 'spell;
            }

            // Wait until the next epoch.
            if rx.recv().await.is_none() {
                // Notifier has dropped the sender. Based on the current implementation of
                // different things, this can happen when the application's execution socket
                // is dropped.
                return;
            }
        }
    });

    w_rx
}

/// Map each topic to a fixed size number.
#[inline(always)]
fn topic_to_index(topic: Topic) -> usize {
    match topic {
        Topic::Consensus => 0,
        Topic::DistributedHashTable => 1,
        Topic::Debug => 2,
    }
}