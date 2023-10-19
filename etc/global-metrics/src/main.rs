mod config;
mod ip_api;
mod types;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{anyhow, Result};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{self, Extension, Json, Router};
use hyper::{Body, Client, Method, Request};
use hyper_tls::HttpsConnector;
use lazy_static::lazy_static;
use lightning_types::NodeInfo;
use moka::sync::Cache;
use resolved_pathbuf::ResolvedPathBuf;
use serde_json::{json, Value};
use sled::Db;
use tracing::{error, info};

use crate::config::Config;
use crate::ip_api::{get_ip_info, IpInfoResponse};
use crate::types::{PrometheusDiscoveryChunk, RpcResponse};

lazy_static! {
    static ref NODE_REGISTRY_REQUEST: serde_json::Value = json!({
        "jsonrpc": "2.0",
        "method":"flk_get_node_registry",
        "params":[],
        "id":1,
    });
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let config = Config::default();
    let path = ResolvedPathBuf::try_from(config.db_path.as_ref()).unwrap();
    let store = sled::open(path).unwrap();
    let cache: Cache<String, IpInfoResponse> = Cache::builder()
        .max_capacity(10_000)
        .time_to_live(Duration::from_secs(72 * 60 * 60))
        .time_to_idle(Duration::from_secs(15 * 60))
        .build();
    let app = Router::new()
        .route("/http_sd", get(service_discovery))
        .layer(Extension(store))
        .layer(Extension(config))
        .layer(Extension(cache));

    let addr = SocketAddr::from(([0, 0, 0, 0], 4000));
    info!("metrics service discovery listening on {addr}");

    axum::Server::bind(&addr)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap();
}

/// handler for HTTP-based service discovery for prometheus
async fn service_discovery(
    Extension(config): Extension<Config>,
    Extension(store): Extension<Db>,
    Extension(cache): Extension<Cache<String, IpInfoResponse>>,
) -> (StatusCode, Json<Value>) {
    let nodes = match get_node_registry(&config).await {
        Ok(n) => n,
        Err(e) => {
            error!("{e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!(e.to_string())),
            );
        },
    };
    let mut batch = sled::Batch::default();
    let mut discovery_chunk = Vec::new();
    for node in nodes {
        let public_key = node.public_key;
        let node_target = format!("{}:{}", node.domain.to_string(), node.ports.rpc.to_string());

        let mut chunk: Option<PrometheusDiscoveryChunk> = match store.get(public_key.0) {
            Ok(Some(value)) => {
                let chunk: PrometheusDiscoveryChunk = bincode::deserialize(&value).unwrap();
                if chunk.get_targets().first().unwrap() == &node_target {
                    Some(chunk)
                } else {
                    None
                }
            },
            _ => None,
        };

        if chunk.is_none() {
            let domain = node.domain.to_string();
            let ip_info = match cache.get(&domain) {
                Some(info) => info,
                None => match get_ip_info(&config.ipinfo_token, node.domain.to_string()).await {
                    Ok(ip_info) => {
                        cache.insert(domain, ip_info.clone());
                        ip_info
                    },
                    Err(e) => {
                        error!("Lookup failed for IP: {}, due to {}", node.domain, e);
                        continue;
                    },
                },
            };

            let targets = vec![node_target];
            let mut labels = HashMap::new();
            labels.insert("public_key".to_string(), node.public_key.to_string());
            labels.insert("geohash".to_string(), ip_info.geo.clone());
            labels.insert("country_code".to_string(), ip_info.country.clone());
            labels.insert("timezone".to_string(), ip_info.timezone.clone());

            let local_chunk = PrometheusDiscoveryChunk::new(targets, labels);
            let chunk_to_bytes = bincode::serialize(&local_chunk).unwrap();
            batch.insert(node.public_key.0.to_vec(), chunk_to_bytes);
            chunk = Some(local_chunk)
        }
        discovery_chunk.push(chunk.unwrap());
    }
    let _ = store.apply_batch(batch);
    (StatusCode::OK, Json(json!(discovery_chunk)))
}

async fn get_node_registry(config: &Config) -> Result<Vec<NodeInfo>> {
    let client = Client::builder().build::<_, hyper::Body>(HttpsConnector::new());

    let address = &config.lgtn_node_address;
    let port = &config.lgtn_node_port;

    let uri = format!("http://{address}:{port}/rpc/v0");

    let req = Request::builder()
        .header("Content-Type", "application/json")
        .method(Method::POST)
        .uri(uri)
        .body(Body::from(NODE_REGISTRY_REQUEST.to_string()))
        .map_err(|_| anyhow!("Failed to build the rpc request"))?;

    let response = client
        .request(req)
        .await
        .map_err(|_| anyhow!("Request to rpc service failed"))?;

    if !response.status().is_success() {
        return Err(anyhow!(format!(
            "Failed to get nodes registry with status: {}",
            response.status()
        )));
    }

    let data = hyper::body::to_bytes(response.into_body()).await?;
    let response: RpcResponse<Vec<NodeInfo>> = serde_json::from_slice(&data)?;
    Ok(response.result)
}
