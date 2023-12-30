//! This module contains the implementation of a directory structure made on top of
//! blake3.

mod hash;
mod types;

pub use hash::hash_directory;
pub use types::*;
