use anyhow::bail;
use log::error;
use tokio::io::Interest;
use tokio::net::UnixStream;

use crate::schema::{EbpfServiceFrame, FileOpen, FileOpenSrc, Pf};
use crate::state::SharedState;

pub struct Connection {
    socket: UnixStream,
    shared_state: SharedState,
}

impl Connection {
    pub fn new(socket: UnixStream, shared_state: SharedState) -> Self {
        Self {
            socket,
            shared_state,
        }
    }

    #[inline]
    async fn file_open_handle(&mut self, message: FileOpen) -> anyhow::Result<()> {
        match message.src {
            FileOpenSrc::Pid(pid) => {
                if message.op == FileOpen::ALLOW {
                    self.shared_state.file_open_allow_pid(pid).await
                } else {
                    self.shared_state.file_open_deny_pid(pid).await
                }
            },
            FileOpenSrc::Bin { inode, dev, rdev } => {
                if message.op == FileOpen::ALLOW {
                    self.shared_state
                        .file_open_allow_binfile(inode, dev, rdev)
                        .await
                } else {
                    self.shared_state
                        .file_open_deny_binfile(inode, dev, rdev)
                        .await
                }
            },
        }
    }

    #[inline]
    async fn pf_handle(&mut self, message: Pf) -> anyhow::Result<()> {
        match message.op {
            Pf::ADD => {
                self.shared_state.packet_filter_add(message.addr).await?;
            },
            Pf::REMOVE => {
                self.shared_state.packet_filter_remove(message.addr).await?;
            },
            op => {
                bail!("invalid op: {op:?}");
            },
        }
        Ok(())
    }

    #[inline]
    async fn handle_request(&mut self, frame: EbpfServiceFrame) -> anyhow::Result<()> {
        match frame {
            EbpfServiceFrame::FileOpen(file_open) => self.file_open_handle(file_open).await,
            EbpfServiceFrame::Pf(pf) => self.pf_handle(pf).await,
        }
    }

    pub async fn handle(mut self) -> anyhow::Result<()> {
        let mut read_buf = vec![0u8; 8];
        let mut bytes_read = 0;
        let mut frame_len = 0;
        loop {
            self.socket.ready(Interest::READABLE).await?;
            'read: loop {
                while frame_len == 0 && bytes_read < 8 {
                    match self.socket.try_read(&mut read_buf[bytes_read..]) {
                        Ok(0) => {
                            return Ok(());
                        },
                        Ok(n) => {
                            bytes_read += n;
                        },
                        Err(e) if e.kind() == tokio::io::ErrorKind::WouldBlock => {
                            // We received a false positive.
                            break 'read;
                        },
                        Err(e) => {
                            return Err(e.into());
                        },
                    }
                }

                if frame_len == 0 {
                    let bytes: [u8; 8] = read_buf.as_slice().try_into()?;
                    frame_len = usize::from_be_bytes(bytes);
                    // We subtract here to pass entire buffer
                    // to EbpfServiceFrame deserializer further below.
                    read_buf.resize(frame_len - 8, 0);
                    bytes_read = 0;
                }

                while bytes_read < frame_len {
                    match self.socket.try_read(&mut read_buf[bytes_read..]) {
                        Ok(0) => {
                            return Ok(());
                        },
                        Ok(n) => {
                            bytes_read += n;
                        },
                        Err(e) if e.kind() == tokio::io::ErrorKind::WouldBlock => {
                            // We received a false positive.
                            break 'read;
                        },
                        Err(e) => {
                            return Err(e.into());
                        },
                    }
                }

                match EbpfServiceFrame::try_from(read_buf.as_slice()) {
                    Ok(f) => {
                        if let Err(e) = self.handle_request(f).await {
                            error!("failed to handle request: {e:?}");
                        }
                    },
                    Err(e) => {
                        error!("failed to deserialize frame: {e:?}");
                    },
                }

                read_buf.resize(8, 0);
                bytes_read = 0;
                frame_len = 0;

                break 'read;
            }
        }
    }
}
