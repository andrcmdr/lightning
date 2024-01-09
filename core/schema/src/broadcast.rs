use fleek_crypto::NodeSignature;
use ink_quill::{ToDigest, TranscriptBuilder};
use lightning_types::{Digest, ImmutablePointer, NodeIndex, Topic};
use serde::{Deserialize, Serialize};

use crate::AutoImplSerde;

pub type MessageInternedId = u16;

/// Once a content is put on the network (i.e a node fetches the content from the origin), the
/// node that fetched the content computes the blake3 hash of the content and signs a record
/// attesting that it witnessed the immutable pointer resolving to the said blake3 hash.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ResolvedImmutablePointerRecord {
    /// The immutable pointer that was fetched.
    pub pointer: ImmutablePointer,
    /// The blake3 hash of the content. Used to store the content on the blockstore.
    pub hash: [u8; 32],
    /// The public key of the node which fetched and attested to this content.
    pub originator: NodeIndex,
    /// The signature of the node.
    pub signature: NodeSignature,
}

impl ToDigest for ResolvedImmutablePointerRecord {
    fn transcript(&self) -> TranscriptBuilder {
        TranscriptBuilder::empty("lightning-resolved-pointer")
            .with("pointer-origin", &self.pointer.origin.to_string())
            .with("pointer-uri", &self.pointer.uri)
            .with("hash", &self.hash)
            .with("originator", &self.originator)
    }
}

impl AutoImplSerde for ResolvedImmutablePointerRecord {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Want {
    pub interned_id: MessageInternedId,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Advr {
    pub interned_id: MessageInternedId,
    pub digest: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub origin: NodeIndex,
    pub signature: NodeSignature,
    pub topic: Topic,
    pub timestamp: u64,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Frame {
    /// Sent by a single node to advertise a message that they have.
    Advr(Advr),
    /// Sent by the requester of the message to the advertiser indicating
    /// that they want this message.
    Want(Want),
    /// An actual broadcast message.
    Message(Message),
}

impl ToDigest for Message {
    fn transcript(&self) -> ink_quill::TranscriptBuilder {
        TranscriptBuilder::empty("FLEEK_BROADCAST_DOMAIN")
            .with("topic", &self.topic)
            .with("payload", &self.payload)
    }
}

impl AutoImplSerde for Frame {}
