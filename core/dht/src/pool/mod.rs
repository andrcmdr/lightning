mod client;
mod lookup;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bytes::Bytes;
pub use client::Client;
use futures::future::Either;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use lightning_interfaces::infu_collection::Collection;
use lightning_interfaces::types::NodeIndex;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{oneshot, Notify};
use tokio::task::JoinHandle;

use crate::network;
use crate::network::{FindResponse, Message, UdpTransport, UnreliableTransport};
use crate::pool::lookup::LookupInterface;
use crate::table::Event;

pub enum Task {
    LookUpValue { hash: u32, respond: ValueRespond },
    LookUpNode { hash: u32, respond: ContactRespond },
    Ping { dst: NodeIndex, timeout: Duration },
}

pub struct WorkerPool<C, L, T>
where
    C: Collection,
    L: LookupInterface,
    T: UnreliableTransport,
{
    /// Our node index.
    us: NodeIndex,
    /// Performs look-ups.
    looker: L,
    /// Socket to send/recv messages over the network.
    socket: T,
    /// Queue for receiving tasks.
    task_queue: Receiver<Task>,
    /// Queue for sending events.
    event_queue: Sender<Event>,
    /// Set of currently running worker tasks.
    ongoing_tasks: FuturesUnordered<JoinHandle<u32>>,
    /// Logical set of ongoing lookup tasks.
    ongoing: HashMap<u32, TaskInfo>,
    /// Maximum pool size.
    max_pool_size: usize,
    /// Shutdown notify.
    shutdown: Arc<Notify>,
    _marker: PhantomData<C>,
}

impl<C, L, T> WorkerPool<C, L, T>
where
    C: Collection,
    L: LookupInterface,
    T: UnreliableTransport,
{
    pub async fn run(&mut self) {
        loop {
            tokio::select! {
                biased;
                _ = self.shutdown.notified() => {
                    tracing::info!("shutting down worker pool");
                    break;
                }
                next = self.task_queue.recv() => {
                    let Some(task) = next else {
                        break;
                    };
                    self.handle_incoming_task(task);
                }
                next = self.socket.recv() => {
                    match next {
                        Ok(message) => {
                            self.handle_message(message);
                        }
                        Err(e) => {
                            tracing::error!("unexpected error from socket: {e:?}");
                            break;
                        }
                    }
                }
                Some(task_result) = self.ongoing_tasks.next() => {
                    match task_result {
                        Ok(id) => self.cleanup(id),
                        Err(e) => {
                            tracing::warn!("failed to clean up task: {e:?}");
                        }
                    };
                }
            }
        }
    }

    /// Try to run the task on a worker.
    fn handle_incoming_task(&mut self, task: Task) {
        // Validate.
        if self.ongoing_tasks.len() > self.max_pool_size {
            // Should we add a buffer?
        }

        match task {
            Task::LookUpValue { hash, respond, .. } => {
                let looker = self.looker.clone();
                let id: u32 = rand::random();
                self.ongoing_tasks.push(tokio::spawn(async move {
                    let value = looker.lookup_value(hash).await;
                    // if the client drops the receiver, there's nothing we can do.
                    let _ = respond.send(value);
                    id
                }));
            },
            Task::LookUpNode { hash, respond } => {
                let looker = self.looker.clone();
                let id: u32 = rand::random();
                self.ongoing_tasks.push(tokio::spawn(async move {
                    let nodes = looker.lookup_contact(hash).await;
                    // if the client drops the receiver, there's nothing we can do.
                    let _ = respond.send(nodes);
                    id
                }));
            },
            Task::Ping { dst, timeout } => {
                self.ping(dst, timeout);
            },
        }
    }

    fn handle_message(&mut self, message: (NodeIndex, Message)) {
        let (from, message) = message;
        let id = message.id();
        let token = message.token();
        let ty = message.ty();
        match ty {
            network::PING_TYPE => {
                self.pong(id, token, from);
            },
            network::PONG_TYPE => {
                self.handle_pong(id, token, from);
            },
            network::STORE_TYPE => {},
            network::FIND_NODE_TYPE => {},
            network::FIND_NODE_RESPONSE_TYPE => {},
            network::FIND_VALUE_TYPE => {},
            network::FIND_VALUE_RESPONSE_TYPE => {},
            ty => {
                tracing::warn!("invalid message type id received: {ty}")
            },
        }
    }

    /// Send a ping.
    fn ping(&mut self, dst: NodeIndex, timeout: Duration) {
        if self.ongoing_tasks.len() > self.max_pool_size {
            // Should we buffer it for later?
        }

        let us = self.us;
        let socket = self.socket.clone();
        let event_queue = self.event_queue.clone();

        // Task state.
        let id = rand::random();
        let token = rand::random();
        let (pong_tx, mut pong_rx) = oneshot::channel();

        // Run task.
        // The task pings the peer at most two times before giving up.
        self.ongoing_tasks.push(tokio::spawn(async move {
            let mut attempts = 2;
            let ponged = loop {
                let message = network::ping(id, token, us);
                if socket.send(message, dst).await.is_err() {
                    break None;
                }

                match futures::future::select(Box::pin(tokio::time::sleep(timeout)), pong_rx).await
                {
                    Either::Left((_, pong_rx_fut)) => {
                        attempts -= 1;
                        if attempts == 0 {
                            break None;
                        }
                        pong_rx = pong_rx_fut;
                    },
                    Either::Right((from, _)) => {
                        break Some(from);
                    },
                }
            };

            match ponged {
                None => {
                    let _ = event_queue.send(Event::Unresponsive { index: dst }).await;
                },
                Some(pong_result) => {
                    if pong_result.is_err() {
                        tracing::warn!("ping task info was dropped unexpectedly");
                        let _ = event_queue.send(Event::Unresponsive { index: dst }).await;
                    } else {
                        let from = pong_result.unwrap();
                        let timestamp = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;
                        let _ = event_queue.send(Event::Pong { from, timestamp }).await;
                    }
                },
            }

            id
        }));

        // Save task state.
        self.ongoing.insert(id, TaskInfo::ping(pong_tx, token, dst));
    }

    /// Send a pong.
    pub fn pong(&self, id: u32, token: u32, dst: NodeIndex) {
        if self.ongoing_tasks.len() > self.max_pool_size {
            // Should we buffer it or ignore pings in the case where we're overwhelmed?
        }

        let message = network::pong(id, token, self.us);
        let socket = self.socket.clone();
        self.ongoing_tasks.push(tokio::spawn(async move {
            if let Err(e) = socket.send(message, dst).await {
                tracing::error!("failed to send pong: {e:?}");
            }
            // Since we don't save any state that needs cleanup for
            // this task, it doesn't matter which id we give it.
            rand::random()
        }));
        // We do not update `ongoing` because we don't expect a response for a pong.
    }

    // Todo: There is more to do here as far as validating the peer.
    // Authentication could be handled here or in the network layer.
    // Peers that send invalid tokens for instance could be punished.
    fn handle_pong(&mut self, id: u32, token: u32, from: NodeIndex) {
        if let Some(info) = self.ongoing.get_mut(&id) {
            let ping_info = info
                .ping_info
                .as_mut()
                .expect("Ping task info to have ping information");
            if ping_info.token == token && ping_info.dst == from {
                if let Some(pong_event_respond) = ping_info.pong_event_respond.take() {
                    let _ = pong_event_respond.send(from);
                } else {
                    tracing::warn!("ignoring a second valid pong from peer {from:?}");
                }
            }
        }
    }

    fn cleanup(&mut self, id: u32) {
        self.ongoing.remove(&id);
    }
}

struct TaskInfo {
    find_response_queue: Option<Sender<FindResponse>>,
    ping_info: Option<PingInfo>,
}

impl TaskInfo {
    pub fn ping(
        pong_respond: oneshot::Sender<NodeIndex>,
        ping_token: u32,
        ping_dst: NodeIndex,
    ) -> Self {
        let ping_info = PingInfo {
            pong_event_respond: Some(pong_respond),
            token: ping_token,
            dst: ping_dst,
        };
        Self {
            find_response_queue: None,
            ping_info: Some(ping_info),
        }
    }
}

struct PingInfo {
    pong_event_respond: Option<oneshot::Sender<NodeIndex>>,
    token: u32,
    dst: NodeIndex,
}

pub type ValueRespond = oneshot::Sender<lookup::Result<Option<Bytes>>>;
pub type ContactRespond = oneshot::Sender<lookup::Result<Vec<NodeIndex>>>;