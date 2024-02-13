use anyhow::{anyhow, Context};
use arrayref::array_ref;
use cid::Cid;
use deno_core::url::Url;
use deno_core::v8::IsolateHandle;
use deno_core::{serde_v8, v8, JsRuntime};
use fn_sdk::connection::Connection;
use fn_sdk::header::TransportDetail;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error, info};

use crate::runtime::Runtime;
use crate::stream::{Origin, Request};

mod runtime;
pub mod stream;

pub(crate) mod params {
    use std::time::Duration;

    pub const HEAP_INIT: usize = 1 << 10;
    pub const HEAP_LIMIT: usize = 50 << 20;
    pub const REQ_TIMEOUT: Duration = Duration::from_secs(15);
    pub const FETCH_BLACKLIST: &[&str] = &["localhost", "127.0.0.1", "::1"];
}

#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    fn_sdk::ipc::init_from_env();

    info!("Initialized POC JS service!");

    let mut listener = fn_sdk::ipc::conn_bind().await;

    // Explicitly initialize the v8 platform on the main thread
    JsRuntime::init_platform(None);

    // To cancel events mid execution.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<IsolateHandle>();
    tokio::spawn(async move {
        while let Some(handle) = rx.recv().await {
            tokio::spawn(async move {
                tokio::time::sleep(params::REQ_TIMEOUT).await;
                handle.terminate_execution();
            });
        }
    });

    while let Ok(conn) = listener.accept().await {
        let tx = tx.clone();

        // spawn a new thread and tokio runtime to handle the connection
        // TODO: This is very hacky and not very scalable
        // Research using deno's JsRealms to provide the script sandboxing in a single or a
        // few shared multithreaded runtimes, or use a custom work scheduler.
        std::thread::spawn(move || {
            if let Err(e) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create connection async runtime")
                .block_on(handle_connection(tx, conn))
            {
                error!("session failed: {e:?}");
            }
        });
    }
}

async fn handle_connection(
    tx: UnboundedSender<IsolateHandle>,
    mut connection: Connection,
) -> anyhow::Result<()> {
    if connection.is_http_request() {
        // url = http://fleek/:origin/:hash
        let body = connection
            .read_payload()
            .await
            .context("Could not read body.")?;
        let TransportDetail::HttpRequest { uri, .. } = &connection.header.transport_detail else {
            unreachable!()
        };
        let request = extract_request(uri, &body).context("Could not parse request")?;
        handle_request(&mut connection, &tx, request).await?;
    } else {
        while let Some(payload) = connection.read_payload().await {
            let request: Request = serde_json::from_slice(&payload)?;
            handle_request(&mut connection, &tx, request).await?;
        }
    }
    Ok(())
}

async fn handle_request(
    connection: &mut Connection,
    tx: &UnboundedSender<IsolateHandle>,
    request: Request,
) -> anyhow::Result<()> {
    let Request {
        origin,
        uri,
        path,
        param,
    } = request;

    // Fetch content from origin
    let hash = match origin {
        Origin::Blake3 => {
            let hash = hex::decode(uri).context("failed to decode blake3 hash")?;

            if hash.len() != 32 {
                connection
                    .write_payload(b"Invalid blake3 hash length")
                    .await
                    .context("failed to send error message")?;
                return Err(anyhow!("invalid blake3 hash length"));
            }

            let hash = *array_ref![hash, 0, 32];

            if fn_sdk::api::fetch_blake3(hash).await {
                hash
            } else {
                connection
                    .write_payload(b"Failed to fetch blake3 content")
                    .await
                    .context("failed to send error message")?;
                return Err(anyhow!("failed to fetch file"));
            }
        },
        Origin::Ipfs => {
            match fn_sdk::api::fetch_from_origin(
                origin.into(),
                Cid::try_from(uri).context("invalid ipfs cid")?.to_bytes(),
            )
            .await
            {
                Some(hash) => hash,
                None => {
                    connection
                        .write_payload(b"Failed to fetch from origin")
                        .await
                        .context("failed to send error message")?;
                    return Err(anyhow!("failed to fetch from origin"));
                },
            }
        },
        o => {
            let err = anyhow!("unknown origin: {o:?}");
            connection
                .write_payload(err.to_string().as_bytes())
                .await
                .context("failed to send error message")?;
            return Err(err);
        },
    };

    let mut location = Url::parse(&format!("blake3://{}", hex::encode(hash)))
        .context("failed to create base url")?;
    if let Some(path) = path {
        location = location.join(&path).context("invalid path string")?;
    }

    // Read and parse the source from the blockstore
    let source_bytes = fn_sdk::blockstore::ContentHandle::load(&hash)
        .await
        .context("failed to get handle for source from blockstore")?
        .read_to_end()
        .await
        .context("failed to read source from blockstore")?;

    let mut source = String::from_utf8(source_bytes).context("failed to parse source as utf8")?;
    // Append entry point function with parameters
    match param {
        Some(param) => source.push_str(&format!("\nmain({param})")),
        None => source.push_str("\nmain()"),
    }

    // Create runtime and execute the source
    let mut runtime = match Runtime::new(location) {
        Ok(runtime) => runtime,
        Err(e) => {
            connection
                .write_payload(e.to_string().as_bytes())
                .await
                .context("failed to send error message")?;
            return Err(e).context("failed to initialize runtime");
        },
    };

    tx.send(runtime.deno.v8_isolate().thread_safe_handle())
        .context("Failed to send the IsolateHandle to main thread.")?;

    let res = match runtime.exec(source) {
        Ok(res) => res,
        Err(e) => {
            connection
                .write_payload(e.to_string().as_bytes())
                .await
                .context("failed to send error message")?;
            return Err(e).context("failed to run javascript");
        },
    };

    // Resolve async if applicable
    // TODO: figure out why `deno.resolve` doesn't drive async functions
    #[allow(deprecated)]
    let res = match tokio::time::timeout(params::REQ_TIMEOUT, runtime.deno.resolve_value(res)).await
    {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => {
            connection
                .write_payload(e.to_string().as_bytes())
                .await
                .context("failed to send error message")?;
            return Err(e).context("failed to resolve output");
        },
        Err(e) => {
            connection
                .write_payload(b"Request timeout")
                .await
                .context("failed to send error message")?;
            return Err(e).context("execution timeout");
        },
    };

    {
        // Handle the return data
        let scope = &mut runtime.deno.handle_scope();
        let local = v8::Local::new(scope, res);

        if local.is_uint8_array() || local.is_array_buffer() {
            // If the return type is a U8 array, send the raw data directly to the client
            let bytes = match deno_core::_ops::to_v8_slice_any(local) {
                Ok(slice) => slice.to_vec(),
                Err(e) => return Err(anyhow!("failed to parse bytes: {e}")),
            };
            connection
                .write_payload(&bytes)
                .await
                .context("failed to send byte response")?;
        } else if local.is_string() {
            // Likewise for string types
            let string = serde_v8::from_v8::<String>(scope, local)
                .context("failed to deserialize response string")?;
            connection
                .write_payload(string.as_bytes())
                .await
                .context("failed to send string response")?;
        } else {
            // Otherwise, send the data as json
            let value = serde_v8::from_v8::<serde_json::Value>(scope, local)
                .context("failed to deserialize response")?
                .clone();
            let res = serde_json::to_string(&value).context("failed to encode json response")?;
            connection
                .write_payload(res.as_bytes())
                .await
                .context("failed to send json response")?;
        }
    }

    let feed = runtime.end();
    debug!("{feed:?}");

    Ok(())
}

fn extract_request(url: &Url, body: &[u8]) -> Option<Request> {
    let mut segments = url.path_segments()?;
    let seg1 = segments.next()?;
    let seg2 = segments.next()?;
    let origin = match seg1 {
        "blake3" => Origin::Blake3,
        "ipfs" => Origin::Ipfs,
        _ => Origin::Unknown,
    };

    let mut path = String::new();
    for s in segments {
        path.push('/');
        path.push_str(s);
    }
    if path.is_empty() {
        path.push('/');
    }
    if let Some(q) = url.query() {
        path.push('?');
        path.push_str(q);
    }
    if let Some(f) = url.fragment() {
        path.push('#');
        path.push_str(f);
    }

    let param = if body.is_empty() {
        url.query_pairs()
            .find_map(|(key, value)| {
                (key == "param").then(|| serde_json::from_str::<serde_json::Value>(&value))
            })
            .transpose()
            .ok()?
    } else {
        // TODO: Send array buffer to JS.
        Some(serde_json::from_slice::<serde_json::Value>(body).ok()?)
    };

    Some(Request {
        origin,
        uri: seg2.to_string(),
        path: Some(path),
        param,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    fn body(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn test_extract_request() {
        assert_eq!(
            extract_request(
                &Url::parse("http://fleek/blake3/content-hash/").unwrap(),
                &[]
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/".to_string()),
                param: None,
            })
        );

        assert_eq!(
            extract_request(
                &Url::parse("http://fleek/blake3/content-hash/a").unwrap(),
                &[]
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/a".to_string()),
                param: None,
            })
        );

        assert_eq!(
            extract_request(
                &Url::parse("http://fleek/blake3/content-hash/a/b").unwrap(),
                &[]
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/a/b".to_string()),
                param: None,
            })
        );

        assert_eq!(
            extract_request(
                &Url::parse("http://fleek/blake3/content-hash/a/b?a=4").unwrap(),
                &[]
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/a/b?a=4".to_string()),
                param: None,
            })
        );

        assert_eq!(
            extract_request(
                &Url::parse("http://fleek/blake3/content-hash/a/b?a=4#hello").unwrap(),
                &[]
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/a/b?a=4#hello".to_string()),
                param: None,
            })
        );

        assert_eq!(
            extract_request(
                &Url::parse(
                    "http://fleek/blake3/content-hash/a/b?a=4&param=%7B%22a%22%3A%204%7D#hello"
                )
                .unwrap(),
                &[]
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/a/b?a=4&param=%7B%22a%22%3A%204%7D#hello".to_string()),
                param: Some(json!({"a": 4})),
            })
        );

        assert_eq!(
            extract_request(
                &Url::parse(
                    "http://fleek/blake3/content-hash/a/b?a=4&param=%7B%22a%22%3A%204%7D#hello"
                )
                .unwrap(),
                &body(json!({"hello": 5}))
            ),
            Some(Request {
                origin: Origin::Blake3,
                uri: "content-hash".to_string(),
                path: Some("/a/b?a=4&param=%7B%22a%22%3A%204%7D#hello".to_string()),
                param: Some(json!({"hello": 5})),
            })
        );
    }
}
