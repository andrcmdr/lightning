use std::cell::RefCell;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use derive_more::AddAssign;
use fleek_crypto::NodePublicKey;
use futures::stream::FuturesUnordered;
use futures::Future;
use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use lightning_interfaces::schema::LightningMessage;
use lightning_interfaces::types::NodeIndex;
use lightning_interfaces::SyncQueryRunnerInterface;
use netkit::endpoint::{NodeAddress, Request};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::ev::Topology;
use crate::frame::{Digest, Frame};
use crate::stats::{ConnectionStats, Stats};
use crate::{Advr, Message, MessageInternedId, Want};

/// This struct is responsible for holding the state of the current peers
/// that we are connected to.
pub struct Peers {
    /// The id of our node.
    us: NodeIndex,
    /// Our access to reporting stats.
    pub stats: Stats,
    /// Peers that are pinned. These are connections suggested by the topology. We
    /// do not drop these connections when performing garbage collection.
    pinned: FxHashSet<NodePublicKey>,
    /// Map each public key to the info we have about that peer.
    peers: im::HashMap<NodePublicKey, Peer>,
    /// Sender for requests to endpoint.
    endpoint_tx: Sender<Request>,
}

impl Peers {
    pub fn new(endpoint_tx: Sender<Request>) -> Self {
        Self {
            us: 0,
            pinned: Default::default(),
            stats: Default::default(),
            peers: Default::default(),
            endpoint_tx,
        }
    }
}

/// An interned id. But not from our interned table.
pub type RemoteInternedId = MessageInternedId;

struct Peer {
    /// The index of the node.
    index: NodeIndex,
    address: SocketAddr,
    has: im::HashMap<MessageInternedId, RemoteInternedId>,
}

impl Clone for Peer {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            address: self.address,
            has: self.has.clone(),
        }
    }
}

/// The originator of a connection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ConnectionOrigin {
    // We have established the connection.
    Us,
    /// The remote has dialed us and we have this connection because we got
    /// a connection from the listener.
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// The connection with the other peer is open.
    ///
    /// We are sending advertisements to this node and actively listening for their
    /// advertisements.
    Open,
    /// The connection with this node is closing. We do not wish to interact with it
    /// anymore. But since we may have pending communication with them. We are still
    /// keeping the connection alive.
    ///
    /// At this point we do not care about their advertisements. We only care about
    /// the messages they owe us. Once the other peer does not owe us anything anymore
    /// we close the connection.
    Closing,
    /// The connection with the other peer is closed and we are not communicating with
    /// the node.
    Closed,
}

impl Peers {
    pub fn get_node_index(&self, pk: &NodePublicKey) -> Option<NodeIndex> {
        self.peers.get(pk).map(|i| i.index)
    }

    pub fn set_current_node_index(&mut self, index: NodeIndex) {
        self.us = index;
    }

    // /// Returns the status of the connection with the given peer. If no connection exists returns
    // /// [`ConnectionStatus::Closed`].
    // pub fn get_connection_status(&self, pk: &NodePublicKey) -> ConnectionStatus {
    //     self.peers
    //         .get(pk)
    //         .map(|e| e.status)
    //         .unwrap_or(ConnectionStatus::Closed)
    // }

    /// Unpin every pinned connection.
    pub fn unpin_all(&mut self) {
        self.pinned.clear();
    }

    /// Pin a peer to prevent garbage collector to remove the connection.
    pub fn pin_peer(&mut self, pk: NodePublicKey) {
        self.pinned.insert(pk);
    }

    /// Move every connection made by us that is not pinned into closing state.
    pub fn disconnect_unpinned(&mut self) {
        // TODO(qti3e)
    }

    /// Insert a mapping from a local interned message id we have to the remote interned id for a
    /// given remote node.
    pub fn insert_index_mapping(
        &mut self,
        remote: &NodePublicKey,
        local_index: MessageInternedId,
        remote_index: RemoteInternedId,
    ) {
        let Some(info) = self.peers.get_mut(remote) else {
            return;
        };

        info.has.insert(local_index, remote_index);
    }

    /// Get the interned id a remote knows a message we know by our local interned id, or `None`.
    pub fn get_index_mapping(
        &self,
        remote: &NodePublicKey,
        local_index: MessageInternedId,
    ) -> Option<RemoteInternedId> {
        self.peers
            .get(remote)
            .and_then(|i| i.has.get(&local_index))
            .copied()
    }

    /// Send a `Frame::Message` to the specific node.
    // TODO(qti3e): Fix double serialization.
    pub fn send_message(&self, remote: &NodePublicKey, frame: Frame) {
        log::trace!("sending want response to {remote}");
        debug_assert!(matches!(frame, Frame::Message(_)));
        let Some(info) = self.peers.get(remote) else {
            return;
        };

        let mut writer = Vec::new();
        if let Err(e) = frame.encode(&mut writer) {
            log::error!("frame encoding failed: {e:?}");
            return;
        };

        let peer_address = NodeAddress {
            socket_address: info.address,
            pk: *remote,
        };
        let endpoint_tx = self.endpoint_tx.clone();
        tokio::spawn(async move {
            if endpoint_tx
                .send(Request::SendMessage {
                    peer: peer_address,
                    message: writer,
                })
                .await
                .is_err()
            {
                tracing::error!("endpoint dropped the channel");
            }
        });
    }

    /// Send a want request to the given node returns `None` if we don't have the node's
    /// sender anymore.
    pub fn send_want_request(
        &self,
        remote: &NodePublicKey,
        remote_index: RemoteInternedId,
    ) -> bool {
        log::trace!("sending want request to {remote} for {remote_index}");
        let Some(info) = self.peers.get(remote) else {
            return false;
        };

        let mut writer = Vec::new();
        if let Err(e) = Frame::Want(Want {
            interned_id: remote_index,
        })
        .encode(&mut writer)
        {
            log::error!("frame encoding failed: {e:?}");
            return false;
        };

        let peer_address = NodeAddress {
            socket_address: info.address,
            pk: *remote,
        };
        let endpoint_tx = self.endpoint_tx.clone();
        tokio::spawn(async move {
            if endpoint_tx
                .send(Request::SendMessage {
                    peer: peer_address,
                    message: writer,
                })
                .await
                .is_err()
            {
                tracing::error!("endpoint dropped the channel");
            }
        });

        true
    }

    /// Advertise a given digest with the given assigned interned id to all the connected
    /// peers that we have. Obviously, we don't do it for the nodes that we already know
    /// have this message.
    pub fn advertise(&self, id: MessageInternedId, digest: Digest) {
        let mut message = Vec::new();
        if Frame::Advr(Advr {
            interned_id: id,
            digest,
        })
        .encode(&mut message)
        .is_err()
        {
            log::error!("failed to encode advertisement");
            return;
        };

        let endpoint_tx = self.endpoint_tx.clone();
        self.for_each(move |stats, (pk, info)| {
            if info.has.contains_key(&id) {
                return;
            }

            stats.report(
                info.index,
                ConnectionStats {
                    advertisements_received_from_us: 1,
                    ..Default::default()
                },
            );

            let sender = endpoint_tx.clone();
            let msg = message.clone();
            tokio::spawn(async move {
                // TODO(qti3e): Explore allowing to send raw buffers from here.
                // There is a lot of duplicated serialization going on because
                // of this pattern.
                //
                // struct Envelop<T>(Vec<u8>, PhantomData<T>);
                // let e = Envelop::new(msg);
                // ...
                // send_envelop(e)
                let address = NodeAddress {
                    socket_address: info.address,
                    pk,
                };
                if sender
                    .send(Request::SendMessage {
                        peer: address,
                        message: msg,
                    })
                    .await
                    .is_err()
                {
                    tracing::error!("endpoint dropped the channel");
                }
            });
        });
    }

    #[inline]
    fn for_each<F>(&self, closure: F)
    where
        F: Fn(&Stats, (NodePublicKey, Peer)) + Send + 'static,
    {
        // Take a snapshot of the state. This is O(1).
        let state = self.peers.clone();
        let stats = self.stats.clone();

        // TODO(qti3e): Find a good spawn threshold.
        if state.len() >= 100 {
            tokio::spawn(async move {
                for (pk, info) in state {
                    closure(&stats, (pk, info));
                }
            });
        } else {
            for (pk, info) in state {
                closure(&stats, (pk, info));
            }
        }
    }

    pub fn handle_new_connection(
        &mut self,
        index: NodeIndex,
        peer_pk: NodePublicKey,
        peer_address: SocketAddr,
    ) {
        assert_ne!(index, self.us); // we shouldn't be calling ourselves.

        if let Some(info) = self.peers.get(&peer_pk) {
            debug_assert_eq!(index, info.index);
            return;
        }

        let info = Peer {
            index,
            address: peer_address,
            has: Default::default(),
        };

        self.peers.insert(peer_pk, info);
    }

    #[inline(always)]
    pub fn handle_disconnect(&mut self, peer: &NodePublicKey) {
        self.peers.remove(&peer);
    }

    #[inline(always)]
    pub fn report_stats(&mut self, peer: NodePublicKey, frame: &Frame) {
        if let Some(info) = self.peers.get(&peer) {
            // Update the stats on what we got.
            self.stats.report(
                info.index,
                match frame {
                    Frame::Advr(_) => ConnectionStats {
                        advertisements_received_from_peer: 1,
                        ..Default::default()
                    },
                    Frame::Want(_) => ConnectionStats {
                        wants_received_from_peer: 1,
                        ..Default::default()
                    },
                    Frame::Message(_) => ConnectionStats {
                        messages_received_from_peer: 1,
                        ..Default::default()
                    },
                },
            );
        }
    }

    /// Based on the number of messages the other node owes us. And the time elapsed since
    /// we decided to close the connection, returns a boolean indicating whether or not we
    /// are interested in keeping this connection going and if we should still keep listening
    /// for a message to come.
    #[inline(always)]
    fn keep_alive(&self, pk: &NodePublicKey) -> bool {
        // TODO(qti3e): Implement this function.
        true
    }
}
