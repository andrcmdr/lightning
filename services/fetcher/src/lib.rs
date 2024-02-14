//! # Fleek Network Fetcher Service
//!
//! ## Request layout:
//!
//! ```text
//! Payload [ origin (u8) . uid (<1024 bytes) ]
//! ```
//!
//! ## Response:
//!
//! Service will send a single u32 counter with the number of blocks for the content.
//! The content will then be streamed in 256KiB payloads.

use anyhow::bail;
use arrayref::array_ref;
use bytes::{Buf, Bytes};
use cid::Cid;
use fn_sdk::api::Origin as ApiOrigin;
use fn_sdk::connection::Connection;
use fn_sdk::header::TransportDetail;
use tracing::{debug, error, info};
use url::Url;

#[derive(Debug)]
#[repr(u8)]
pub enum Origin {
    Blake3 = 0x00,
    IPFS = 0x01,
    Unknown = 0xFF,
}

impl From<u8> for Origin {
    #[inline(always)]
    fn from(val: u8) -> Self {
        match val {
            0 => Self::Blake3,
            1 => Self::IPFS,
            _ => Self::Unknown,
        }
    }
}

impl From<Origin> for ApiOrigin {
    #[inline(always)]
    fn from(val: Origin) -> Self {
        match val {
            Origin::IPFS => ApiOrigin::IPFS,
            _ => unreachable!(),
        }
    }
}

#[tokio::main]
pub async fn main() {
    fn_sdk::ipc::init_from_env();
    info!("Initialized IPFS fetcher service!");

    let mut listener = fn_sdk::ipc::conn_bind().await;
    while let Ok(conn) = listener.accept().await {
        tokio::spawn(handle_connection(conn));
    }
}

pub async fn handle_connection(mut conn: Connection) {
    debug!("new connection");
    if conn.is_http_request() {
        let TransportDetail::HttpRequest { uri, .. } = &conn.header.transport_detail else {
            unreachable!()
        };
        let Some((origin, uri)) = parse_http_url(uri) else {
            let _ = conn.write_payload(b"invalid request url").await;
            return;
        };
        if let Err(e) = handle_request(&mut conn, origin, uri).await {
            error!("{e}");
        }
    } else {
        while let Some(mut payload) = conn.read_payload().await {
            let origin = Origin::from(payload[0]);
            payload.advance(1);
            if let Err(e) = handle_request(&mut conn, origin, payload.into()).await {
                error!("{e}");
            }
        }
    }
}

fn parse_http_url(url: &Url) -> Option<(Origin, Bytes)> {
    let mut segments = url.path_segments()?;
    let seg1 = segments.next()?;
    let seg2 = segments.next()?;
    let origin = match seg1 {
        "blake3" => Origin::Blake3,
        "ipfs" => Origin::IPFS,
        _ => return None,
    };
    let uri = match origin {
        Origin::Blake3 => hex::decode(seg2).ok()?,
        Origin::IPFS => Cid::try_from(seg2).ok()?.into(),
        Origin::Unknown => unreachable!(),
    };
    Some((origin, uri.into()))
}

async fn handle_request(conn: &mut Connection, origin: Origin, uri: Bytes) -> anyhow::Result<()> {
    debug!("got request for cid");

    // Fetch the content from the origin
    let hash = match origin {
        Origin::Unknown => {
            bail!("unknown origin");
        },
        Origin::Blake3 => {
            if uri.len() != 32 {
                bail!("expected a 32 byte hash");
            }

            // Fetch the content from the network
            let hash = *array_ref!(uri, 0, 32);
            if !fn_sdk::api::fetch_blake3(hash).await {
                bail!("failed to fetch content");
            }

            hash
        },
        origin => {
            // Fetch the content from the origin
            let Some(hash) = fn_sdk::api::fetch_from_origin(origin.into(), uri).await else {
                bail!("failed to fetch from origin");
            };
            hash
        },
    };

    debug!("downloaded content");

    // Get the content from the blockstore
    let Ok(content_handle) = fn_sdk::blockstore::ContentHandle::load(&hash).await else {
        bail!("failed to load content handle from the blockstore");
    };

    debug!("got content handle");

    // Only write block count for non-HTTP transports.
    if !conn.is_http_request() {
        let bytes = (content_handle.len() as u32).to_be_bytes();
        if let Err(e) = conn.write_payload(bytes.as_slice()).await {
            bail!("failed to send number of blocks: {e}");
        }
        debug!("sent block count {}", content_handle.len());
    }

    for block in 0..content_handle.len() {
        let Ok(bytes) = content_handle.read(block).await else {
            bail!("failed to read content from the blockstore :(");
        };

        debug!("sending block {block}");

        if let Err(e) = conn.write_payload(&bytes).await {
            bail!("failed to send block: {e}");
        }
    }

    Ok(())
}
