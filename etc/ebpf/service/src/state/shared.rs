use std::net::SocketAddrV4;
use std::sync::Arc;

use anyhow::bail;
use aya::maps::{HashMap, MapData};
use common::{File, PacketFilter};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::state::schema::Filter;

const EBPF_STATE_TMP_DIR: &str = "~/.lightning/ebpf/state/tmp";
const EBPF_STATE_PF_DIR: &str = "~/.lightning/ebpf/state/packet-filter";
const RULES_FILE: &str = "rules.json";

#[derive(Clone)]
pub struct SharedState {
    packet_filters: Arc<Mutex<HashMap<MapData, PacketFilter, u32>>>,
    pid_open_file_allow: Arc<Mutex<HashMap<MapData, u64, u64>>>,
    binfile_open_file_allow: Arc<Mutex<HashMap<MapData, File, u64>>>,
    pid_open_file_deny: Arc<Mutex<HashMap<MapData, u64, u64>>>,
    binfile_open_file_deny: Arc<Mutex<HashMap<MapData, File, u64>>>,
}

impl SharedState {
    pub fn new(
        packet_filters: HashMap<MapData, PacketFilter, u32>,
        pid_open_file_allow: HashMap<MapData, u64, u64>,
        binfile_to_file_allow: HashMap<MapData, File, u64>,
        pid_open_file_deny: HashMap<MapData, u64, u64>,
        binfile_to_file_deny: HashMap<MapData, File, u64>,
    ) -> Self {
        Self {
            packet_filters: Arc::new(Mutex::new(packet_filters)),
            pid_open_file_allow: Arc::new(Mutex::new(pid_open_file_allow)),
            binfile_open_file_allow: Arc::new(Mutex::new(binfile_to_file_allow)),
            pid_open_file_deny: Arc::new(Mutex::new(pid_open_file_deny)),
            binfile_open_file_deny: Arc::new(Mutex::new(binfile_to_file_deny)),
        }
    }

    pub async fn packet_filter_add(&mut self, addr: SocketAddrV4) -> anyhow::Result<()> {
        let mut map = self.packet_filters.lock().await;
        map.insert(
            PacketFilter {
                ip: (*addr.ip()).into(),
                port: addr.port() as u32,
            },
            0,
            0,
        )?;
        self.sync_packet_filters().await
    }

    pub async fn packet_filter_remove(&mut self, addr: SocketAddrV4) -> anyhow::Result<()> {
        let mut map = self.packet_filters.lock().await;
        map.remove(&PacketFilter {
            ip: (*addr.ip()).into(),
            port: addr.port() as u32,
        })?;
        self.sync_packet_filters().await
    }

    pub async fn file_open_allow_pid(&mut self, pid: u64) -> anyhow::Result<()> {
        let mut map = self.pid_open_file_allow.lock().await;
        map.insert(pid, 0, 0)?;
        Ok(())
    }

    pub async fn file_open_deny_pid(&mut self, pid: u64) -> anyhow::Result<()> {
        let mut map = self.pid_open_file_deny.lock().await;
        map.remove(&pid)?;
        Ok(())
    }

    pub async fn file_open_allow_binfile(
        &mut self,
        inode: u64,
        dev: u32,
        rdev: u32,
    ) -> anyhow::Result<()> {
        let mut map = self.binfile_open_file_allow.lock().await;
        map.insert(
            File {
                inode_n: inode,
                dev,
                rdev,
            },
            0,
            0,
        )?;
        Ok(())
    }

    pub async fn file_open_deny_binfile(
        &mut self,
        inode: u64,
        dev: u32,
        rdev: u32,
    ) -> anyhow::Result<()> {
        let mut map = self.binfile_open_file_deny.lock().await;
        map.remove(&File {
            inode_n: inode,
            dev,
            rdev,
        })?;
        Ok(())
    }

    async fn sync_packet_filters(&self) -> anyhow::Result<()> {
        let tmp_path = format!("{EBPF_STATE_TMP_DIR}/{RULES_FILE}");
        let mut tmp = fs::File::create(tmp_path.clone()).await?;
        let mut rules = Vec::new();
        let guard = self.packet_filters.lock().await;
        for key in guard.keys() {
            match key {
                Ok(key) => {
                    rules.push(Filter {
                        ip: key.ip.into(),
                        port: key.port as u16,
                    });
                },
                Err(e) => bail!("failed to sync blocklist: {e:?}"),
            }
        }

        let bytes = serde_json::to_string(&rules)?;
        tmp.write_all(bytes.as_bytes()).await?;

        tmp.sync_all().await?;

        fs::rename(tmp_path, format!("{EBPF_STATE_PF_DIR}/{RULES_FILE}")).await?;

        Ok(())
    }
}
