use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub reporter_buffer_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            reporter_buffer_size: 1,
        }
    }
}
