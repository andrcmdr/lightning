use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use fleek_crypto::NodePublicKey;
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;
use tokio::select;
use tokio::sync::mpsc::Sender;
use tokio::sync::{oneshot, Notify};

use crate::network::sock;
use crate::node::NodeInfo;
use crate::store::StoreRequest;
use crate::table::server::{TableKey, TableRequest};
use crate::task::ResponseEvent;

pub async fn start_worker(
    response_queue_tx: Sender<ResponseEvent>,
    table_tx: Sender<TableRequest>,
    store_tx: Sender<StoreRequest>,
    socket: Arc<UdpSocket>,
    local_key: NodePublicKey,
    shutdown_notify: Arc<Notify>,
) {
    let mut handler = NetworkReceiver {
        response_queue_tx,
        local_key,
        table_tx,
        store_tx,
        socket: socket.clone(),
    };
    loop {
        select! {
            incoming = sock::recv_from(&socket) => {
                match incoming {
                    Ok((datagram, address)) => {
                        if let Err(e) = handler.handle_incoming(datagram, address) {
                            tracing::error!("failed to handle incoming datagram: {e:?}");
                        }
                    }
                    Err(e) => {
                        tracing::error!("unexpected error when reading from socket: {e:?}")
                    }
                }
            }
            _ = shutdown_notify.notified() => {
                tracing::info!("shutting down network receiver worker");
                break;
            }
        }
    }
}

async fn handle_query(
    table_tx: Sender<TableRequest>,
    store_tx: Sender<StoreRequest>,
    socket: Arc<UdpSocket>,
    local_key: NodePublicKey,
    message: Message,
    address: SocketAddr,
) -> Result<()> {
    let query: Request = bincode::deserialize(&message.payload)?;
    match query {
        Request::Find { find_value, target } => {
            let (tx, rx) = oneshot::channel();
            table_tx
                .send(TableRequest::ClosestNodes {
                    target,
                    respond: tx,
                })
                .await
                .expect("table worker to not drop the channel");
            let nodes = rx.await.expect("table worker to not drop the channel")?;
            let value = match find_value {
                true => {
                    let (get_tx, get_rx) = oneshot::channel();
                    store_tx
                        .send(StoreRequest::Get {
                            key: target,
                            respond: get_tx,
                        })
                        .await
                        .expect("store worker not to drop channel");
                    get_rx.await.expect("store worker not to drop channel")
                },
                false => None,
            };
            let payload = bincode::serialize(&Response { nodes, value })?;
            let response = Message {
                ty: MessageType::Response,
                token: message.token,
                id: message.id,
                sender_key: local_key,
                payload,
            };
            let bytes = bincode::serialize(&response)?;
            sock::send_to(&socket, bytes.as_slice(), address).await?;

            tracing::trace!(
                "discovered new contact {:?} {:?}",
                message.sender_key,
                address
            );
            table_tx
                .send(TableRequest::AddNode {
                    node: NodeInfo {
                        address,
                        // Todo: Fix this.
                        index: 0,
                        key: message.sender_key,
                        last_responded: None,
                    },
                    respond: None,
                })
                .await
                .expect("table worker to not drop the channel");
        },
        Request::Store { key, value } => {
            store_tx
                .send(StoreRequest::Put { key, value })
                .await
                .expect("store worker not to drop channel");
        },
        Request::Ping => {
            let payload = bincode::serialize(&Response {
                nodes: Vec::new(),
                value: None,
            })?;
            let response = Message {
                ty: MessageType::Response,
                id: message.id,
                token: message.token,
                sender_key: local_key,
                payload,
            };
            let bytes = bincode::serialize(&response)?;
            sock::send_to(&socket, bytes.as_slice(), address).await?;
        },
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
pub enum Request {
    Find { find_value: bool, target: TableKey },
    // Todo: This may not fit on a datagram
    // but we will delegate this task to an
    // encrypted channel.
    Store { key: TableKey, value: Vec<u8> },
    Ping,
}

#[repr(u8)]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub enum MessageType {
    Query = 0x01 << 0,
    Response = 0x01 << 1,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub ty: MessageType,
    // Channel on which to route the response.
    pub id: u64,
    // Random value used that must be returned in response.
    pub token: u64,
    // Sender's public key.
    pub sender_key: NodePublicKey,
    // Payload of message.
    pub payload: Vec<u8>,
}

// Todo: Create some chunking strategy
// to avoid sending datagrams larger than 512.
#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    pub nodes: Vec<NodeInfo>,
    pub value: Option<Vec<u8>>,
}

struct NetworkReceiver {
    response_queue_tx: Sender<ResponseEvent>,
    local_key: NodePublicKey,
    table_tx: Sender<TableRequest>,
    store_tx: Sender<StoreRequest>,
    socket: Arc<UdpSocket>,
}

impl NetworkReceiver {
    fn handle_incoming(&mut self, datagram: Vec<u8>, address: SocketAddr) -> Result<()> {
        let message: Message = bincode::deserialize(datagram.as_slice())?;
        match message.ty {
            MessageType::Query => {
                tokio::spawn(handle_query(
                    self.table_tx.clone(),
                    self.store_tx.clone(),
                    self.socket.clone(),
                    self.local_key,
                    message,
                    address,
                ));
            },
            MessageType::Response => {
                // This should provide some protection against unrequested replies.
                let response: Response = bincode::deserialize(&message.payload)?;
                let event_tx = self.response_queue_tx.clone();
                tokio::spawn(async move {
                    if event_tx
                        .send(ResponseEvent {
                            id: message.id,
                            token: message.token,
                            sender_key: message.sender_key,
                            response,
                        })
                        .await
                        .is_err()
                    {
                        tracing::error!("failed to send response to lookup task")
                    }
                });
            },
        }
        Ok(())
    }
}
