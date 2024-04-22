use std::collections::HashMap;
use std::fmt::Display;
use std::str::FromStr;

use axum::body::Body;
use axum::extract::{OriginalUri, Path, Query};
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use bytes::Bytes;
use fleek_crypto::{ClientPublicKey, ClientSignature};
use fn_sdk::header::{HttpMethod, HttpOverrides, TransportDetail};
use lightning_interfaces::ExecutorProviderInterface;
use lightning_metrics::increment_counter;
use lightning_schema::handshake::{HandshakeRequestFrame, RequestFrame};
use tokio::sync::oneshot;
use url::Url;

use crate::handshake::Context;
use crate::transports::http::{HttpReceiver, HttpSender, Service};

pub async fn handler<P: ExecutorProviderInterface>(
    method: Method,
    OriginalUri(uri): OriginalUri,
    Path((service_id, path)): Path<(String, String)>,
    Query(params): Query<HashMap<String, String>>,
    Extension(provider): Extension<Context<P>>,
    payload: Bytes,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let id = u32::from_str(&service_id)
        .map_err(|_| (StatusCode::NOT_FOUND, "route not found".to_string()))?;

    let method = match method {
        Method::GET => HttpMethod::Get,
        Method::POST => HttpMethod::Post,
        Method::PUT => HttpMethod::Put,
        Method::DELETE => HttpMethod::Delete,
        _ => return Err((StatusCode::NOT_FOUND, "invalid method".to_string())),
    };

    let body_frame = RequestFrame::ServicePayload { bytes: payload };

    let handshake_frame = HandshakeRequestFrame::Handshake {
        service: id,
        pk: ClientPublicKey([0; 96]),
        pop: ClientSignature([0; 48]),
        retry: None,
    };

    let (frame_tx, frame_rx) = async_channel::bounded(8);
    // Todo: Fix synchronization between data transfer and connection proxy.
    // Notes: Reducing the size of this queue produces enough backpressure to slow down the
    // transfer of data and if the socket is dropped before all the data is transferred,
    // the proxy drops the connection and the client receives incomplete data.
    let (body_tx, body_rx) = async_channel::bounded(2_000_000);
    let (termination_tx, termination_rx) = oneshot::channel();

    let sender = HttpSender::new(Service::Js, frame_tx, body_tx, termination_tx);
    let receiver = HttpReceiver::new(
        frame_rx,
        TransportDetail::HttpRequest {
            method,
            uri: extract_url(&path, uri),
            // TODO
            header: Default::default(),
        },
    );

    sender.frame_tx.try_send(Some(body_frame)).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected error".to_string(),
        )
    })?;

    increment_counter!(
        "handshake_http_sessions",
        Some("Counter for number of handshake sessions accepted over http")
    );

    provider
        .handle_new_connection(handshake_frame, sender, receiver)
        .await;
    let header_bytes = body_rx
        .recv()
        .await
        .map_err(|_| bad_request("Connection closed before headers were sent"))
        .and_then(|res| {
            res.map_err(|e| bad_request(format!("Unable to get headers from service: {e}")))
        })?;
    let header_overrides =
        serde_json::from_slice::<HttpOverrides>(&header_bytes).unwrap_or_default();

    let mut response_builder = Response::builder();

    if let Some(content_type) = params.get("mime") {
        response_builder = response_builder.header("Content-Type", content_type);
    }
    if let Some(headers) = header_overrides.headers {
        for header in headers {
            response_builder = response_builder.header(header.0, header.1);
        }
    }
    if let Some(status) = header_overrides.status {
        response_builder = response_builder.status(status);
    }

    let body = Body::from_stream(body_rx);

    // If there is an error while streaming, the status header has already been sent,
    // this is a hacky way of returning an error status before beginning streaming the body.
    match termination_rx.await {
        Ok(reason) => Err(bad_request(format!("handshake failed: {reason:?}"))),
        Err(_) => response_builder
            .body(body)
            .map_err(|_| bad_request("invalid type value")),
    }
}

#[inline(always)]
fn bad_request<T: AsRef<str> + Display>(msg: T) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, msg.to_string())
}

#[inline(always)]
fn extract_url(path: &str, url: Uri) -> Url {
    let mut result = Url::parse("http://fleek/").unwrap();
    result.set_path(path);
    result.set_query(url.query());
    result
}
