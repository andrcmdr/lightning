use core::arch;
use std::ops::Deref;

use derive_more::IsVariant;
use lightning_schema::LightningMessage;
use rkyv::ser::Serializer;
use rkyv::{Archive, Deserialize, Serialize};

use crate::ReqRes;

/// Client public key bytes. We use this type alias instead of the actual fleek-crypto type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct ClientPublicKeyBytes([u8; 96]);

impl From<[u8; 96]> for ClientPublicKeyBytes {
    fn from(value: [u8; 96]) -> Self {
        Self(value)
    }
}

impl From<ClientPublicKeyBytes> for [u8; 96] {
    fn from(value: ClientPublicKeyBytes) -> Self {
        value.0
    }
}

pub type RequestCtxU64 = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct StaticVec<const CAP: usize> {
    size: usize,
    buffer: [u8; CAP],
}

impl<const CAP: usize> Deref for StaticVec<CAP> {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.buffer[0..self.size]
    }
}

impl<const CAP: usize> StaticVec<CAP> {
    pub fn new(src: &[u8]) -> Self {
        let size = src.len();
        assert!(size <= CAP);
        let mut buffer = [0; CAP];
        buffer[0..size].copy_from_slice(src);
        Self { size, buffer }
    }
}
impl<const CAP: usize> From<&StaticVec<CAP>> for Vec<u8> {
    fn from(value: &StaticVec<CAP>) -> Self {
        let mut vec = Vec::with_capacity(value.size);
        vec.extend_from_slice(&value[0..value.size]);
        vec
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct IpcRequest {
    /// A pointer to the request context.
    pub request_ctx: Option<RequestCtxU64>,
    /// The request to be processed by core.
    pub request: Request,
}

/// A message sent from the core process to the service.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum IpcMessage {
    Response {
        request_ctx: RequestCtxU64,
        response: Response,
    },
}

impl LightningMessage for IpcMessage {
    fn decode(buffer: &[u8]) -> anyhow::Result<Self> {
        let archived = rkyv::check_archived_root::<Self>(buffer)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        Ok(archived.deserialize(&mut rkyv::Infallible)?)
    }

    fn encode<W: std::io::prelude::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let mut ser = rkyv::ser::serializers::WriteSerializer::new(writer);

        Ok(ser.serialize_value(self)?)
    }

    fn encode_length_delimited<W: std::io::prelude::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<usize> {
        let encoded = rkyv::to_bytes::<_, 256>(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let len = encoded.len().to_le_bytes();

        writer.write_all(&len)?;
        writer.write_all(&encoded)?;

        Ok(encoded.len() + 8)
    }
}

impl LightningMessage for IpcRequest {
    fn decode(buffer: &[u8]) -> anyhow::Result<Self> {
        let archived = rkyv::check_archived_root::<Self>(buffer)
            .map_err(|e| anyhow::anyhow!(format!("failed to check root {}", e)))?;

        Ok(archived.deserialize(&mut rkyv::Infallible)?)
    }

    fn encode<W: std::io::prelude::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let mut ser = rkyv::ser::serializers::WriteSerializer::new(writer);

        Ok(ser.serialize_value(self)?)
    }

    fn encode_length_delimited<W: std::io::prelude::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<usize> {
        let encoded = rkyv::to_bytes::<_, 256>(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let len = encoded.len().to_le_bytes();

        writer.write_all(&len)?;
        writer.write_all(&encoded)?;

        Ok(encoded.len() + 8)
    }
}

// Recursive expansion of ReqRes! macro
// =====================================

// #[rustfmt::skip]
// #[derive(Clone,Copy,Debug,IsVariant,PartialEq,Eq, ::rkyv::Archive, ::rkyv::Serialize,
// ::rkyv::Deserialize)] #[archive(check_bytes)]
// #[non_exhaustive]
// pub enum Request {
//     #[doc = r" Query a client's bandwidth balance."]
//     QueryClientBandwidth {
//         #[doc = r" The public key of the user that we want their balance."]
//         pk:ClientPublicKeyBytes
//     },#[doc = r" Query a client's FLK balance."]
//     QueryClientFLK {
//         #[doc = r" The public key of the user."]
//         pk:ClientPublicKeyBytes
//     },FetchFromOrigin {
//         origin:u8,#[doc = r" The encoded URI."]
//         uri:StaticVec<256>
//     },#[doc = r" Fetch a content via a blake3 digest."]
//     FetchBlake3 {
//         #[doc = r" Hash of the content we are interested to fetch."]
//         hash:[u8;
//         32]
//     }
// }

// #[rustfmt::skip]
// #[derive(Clone,Copy,Debug,IsVariant,PartialEq,Eq, ::rkyv::Archive, ::rkyv::Serialize,
// ::rkyv::Deserialize)] #[archive(check_bytes)]
// #[non_exhaustive]
// pub enum Response {
//     QueryClientBandwidth {
//         #[doc = r" The balance of the user."]
//         balance:u128
//     },QueryClientFLK {
//         #[doc = r" The balance of the user."]
//         balance:u128
//     },FetchFromOrigin {
//         #[doc = r" Returns the hash of the content on successful fetch."]
//         hash:Option<[u8;
//         32]>
//     },FetchBlake3 {
//         #[doc = r" Returns true if the fetch succeeded."]
//         succeeded:bool
//     }
// }

ReqRes! {
    /// Query a client's bandwidth balance.
    QueryClientBandwidth {
        /// The public key of the user that we want their balance.
        pk: ClientPublicKeyBytes,
        =>
        /// The balance of the user.
        balance: u128,
    },
    /// Query a client's FLK balance.
    QueryClientFLK {
        /// The public key of the user.
        pk: ClientPublicKeyBytes,
        =>
        /// The balance of the user.
        balance: u128,
    },
    FetchFromOrigin {
        origin: u8,
        /// The encoded URI.
        uri: StaticVec<256>,
        =>
        /// Returns the hash of the content on successful fetch.
        hash: Option<[u8; 32]>,
    },
    /// Fetch a content via a blake3 digest.
    FetchBlake3 {
        /// Hash of the content we are interested to fetch.
        hash: [u8; 32],
        =>
        /// Returns true if the fetch succeeded.
        succeeded: bool
    },
}
