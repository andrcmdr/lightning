//! Sometimes we need to treat a flat slice of bytes conceptually as a list of hash digests all
//! after one another. The [`FlatHashSlice`] in this module is just the right wrapper around our
//! slice that provides basic functionality needed for this purpose.
//!
//! This includes easy iteration over each 32-byte of the data, sanity check that a slice
//! represented in this method is indeed a multiple 32-bytes and easier indexing operations.
//!
//! The supported types to go into a valid [`FlatHashSlice`] are `&[[u8; 32]]` as well as
//! conversion from `&[u8]`. In the later case you should use the `try_from` function as
//! this operation could fail.

use std::fmt::Debug;
use std::ops::Index;

use super::error::CollectionTryFromError;
use crate::utils::{flatten, Digest};

const BYTES_POW_2: usize = 5; // 2 ** 5 == 32

/// A flat slice of hashes. This wrapper guarantees that the size of the underlying slice is a
/// multiple of 32.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlatHashSlice<'s> {
    slice: &'s [u8],
}

/// An iterator over a [`FlatHashSlice`] obtained by calling the [iter] method on the slice.
///
/// [iter]: FlatHashSlice::iter
pub struct FlatHashSliceIter<'s> {
    inner: FlatHashSlice<'s>,
}

impl<'s> TryFrom<&'s [u8]> for FlatHashSlice<'s> {
    type Error = CollectionTryFromError;

    #[inline]
    fn try_from(value: &'s [u8]) -> Result<Self, Self::Error> {
        if value.len() & 31 == 0 {
            Ok(FlatHashSlice { slice: value })
        } else {
            Err(CollectionTryFromError::InvalidNumberOfBytes)
        }
    }
}

impl<'s> From<&'s [[u8; 32]]> for FlatHashSlice<'s> {
    #[inline(always)]
    fn from(value: &'s [[u8; 32]]) -> Self {
        Self {
            slice: flatten(value),
        }
    }
}

impl<'s> Index<usize> for FlatHashSlice<'s> {
    type Output = [u8; 32];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("Out of bound.")
        }
        arrayref::array_ref![self.slice, index << BYTES_POW_2, 32]
    }
}

impl<'s> IntoIterator for FlatHashSlice<'s> {
    type Item = &'s [u8; 32];
    type IntoIter = FlatHashSliceIter<'s>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        FlatHashSliceIter { inner: self }
    }
}

impl<'s> AsRef<[u8]> for FlatHashSlice<'s> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.slice
    }
}

impl<'s> FlatHashSlice<'s> {
    #[inline]
    pub fn get(&self, index: usize) -> &'s [u8; 32] {
        if index >= self.len() {
            panic!("Out of bound.")
        }
        arrayref::array_ref![self.slice, index << BYTES_POW_2, 32]
    }

    /// Returns the number of hashes in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.slice.len() >> BYTES_POW_2
    }

    /// Returns true if there are no hashes in this slice.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Returns a reference to the raw slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.slice
    }

    /// Skip the first `n` items from the current slice.
    #[inline]
    pub fn skip(&self, n: usize) -> Self {
        if n > self.len() {
            panic!("Out of bound.");
        }

        Self {
            slice: &self.slice[(n << BYTES_POW_2)..],
        }
    }

    /// Skip the last `n` items from the current slice.
    #[inline]
    pub fn skip_end(&self, n: usize) -> Self {
        if n > self.len() {
            panic!("Out of bound.");
        }

        let n = self.len() - n;
        Self {
            slice: &self.slice[..(n << BYTES_POW_2)],
        }
    }

    /// Returns an iterator over the current hash slice.
    #[inline]
    pub fn iter(&self) -> FlatHashSliceIter<'_> {
        FlatHashSliceIter { inner: *self }
    }
}

impl<'s> Iterator for FlatHashSliceIter<'s> {
    type Item = &'s [u8; 32];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.is_empty() {
            return None;
        }
        let output = self.inner.get(0);
        self.inner = self.inner.skip(1);
        Some(output)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.inner.len();
        (n, Some(n))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner = self.inner.skip(n.min(self.inner.len()));
        self.next()
    }
}

impl<'s> DoubleEndedIterator for FlatHashSliceIter<'s> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.inner.is_empty() {
            return None;
        }
        let output = self.inner.get(self.inner.len() - 1);
        self.inner = self.inner.skip_end(1);
        Some(output)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.inner = self.inner.skip_end(n.min(self.inner.len()));
        self.next_back()
    }
}

impl<'s> Debug for FlatHashSlice<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fmt = f.debug_list();
        fmt.entries(self.iter().map(Digest));
        fmt.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let v = vec![[0u8; 32]; 100];
        let s = FlatHashSlice::from(v.as_slice());
        assert_eq!(s.len(), 100);
    }

    #[test]
    fn iter_simple() {
        let expected = (0..10).map(|i| [i; 32]).collect::<Vec<[u8; 32]>>();
        let actual = FlatHashSlice::from(expected.as_slice())
            .into_iter()
            .map(|h| *h)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn iter_nth() {
        let v = (0..10).map(|i| [i; 32]).collect::<Vec<[u8; 32]>>();
        assert_eq!(
            FlatHashSlice::from(v.as_slice())
                .into_iter()
                .nth(0)
                .unwrap(),
            &[0; 32]
        );
        let mut iter = FlatHashSlice::from(v.as_slice()).into_iter();
        iter.nth(0);
        assert_eq!(iter.nth(0).unwrap(), &[1; 32]);
        assert_eq!(FlatHashSlice::from(v.as_slice()).into_iter().nth(10), None);
        assert_eq!(FlatHashSlice::from(v.as_slice()).into_iter().nth(20), None);
        assert_eq!(
            FlatHashSlice::from(v.as_slice())
                .into_iter()
                .nth(9)
                .unwrap(),
            &[9; 32]
        );
    }

    #[test]
    fn iter_rev() {
        let mut expected = (0..10).map(|i| [i; 32]).collect::<Vec<[u8; 32]>>();
        let actual = FlatHashSlice::from(expected.as_slice())
            .into_iter()
            .rev()
            .map(|h| *h)
            .collect::<Vec<_>>();
        expected.reverse();
        assert_eq!(actual, expected);
    }
}
