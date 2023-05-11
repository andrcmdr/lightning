use serde::{de::DeserializeOwned, Serialize};

pub trait SerdeBackend {
    fn serialize<T>(value: &T) -> Vec<u8>
    where
        T: Serialize;

    fn deserialize<T>(slice: &[u8]) -> T
    where
        T: DeserializeOwned;
}

pub struct BincodeSerde;

impl SerdeBackend for BincodeSerde {
    fn serialize<T>(value: &T) -> Vec<u8>
    where
        T: Serialize,
    {
        bincode::serialize(value).unwrap()
    }

    fn deserialize<T>(slice: &[u8]) -> T
    where
        T: DeserializeOwned,
    {
        bincode::deserialize(slice).unwrap()
    }
}
