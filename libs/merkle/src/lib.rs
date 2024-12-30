use pyo3::prelude::*;

use blake3::{Hash as Blake3Hash, Hasher};
use dusk_merkle::{Aggregate, Tree};
use hex;

const EMPTY_HASH: Item = Item([0; 32]);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Item([u8; 32]);

impl From<Blake3Hash> for Item {
    fn from(h: Blake3Hash) -> Self {
        Self(h.into())
    }
}

impl<const A: usize> Aggregate<A> for Item {
    const EMPTY_SUBTREE: Self = EMPTY_HASH;

    fn aggregate(items: [&Self; A]) -> Self {
        let mut hasher = Hasher::new();
        for item in items {
            hasher.update(&item.0);
        }
        hasher.finalize().into()
    }
}

impl Item {
    #[must_use]
    pub fn new(bytes: [u8; 32]) -> Self {
        Item(bytes)
    }
}

/// Creates a Merkle tree from a list of floating point values and returns its root hash.
///
/// Each value is converted to bytes and hashed using BLAKE3 before being inserted into
/// the tree. The tree has a height of 32 and an arity of 2 (binary tree).
///
/// # Arguments
///
/// * `values` - A vector of f64 values to be inserted into the Merkle tree
///
/// # Returns
///
/// A hexadecimal string prefixed with "0x" representing the Merkle root hash
#[pyfunction]
pub fn insert_values(values: Vec<f64>) -> String {
    const H: usize = 32;
    const A: usize = 2;

    let mut tree = Tree::<Item, H, A>::new();

    for (pos, value) in values.iter().enumerate() {
        let hash_bytes = value.to_be_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&hash_bytes);
        let hash: Item = hasher.finalize().into();

        tree.insert(pos as u64, hash);
    }
    format!("0x{}", hex::encode(tree.root().0))
}

#[pymodule]
fn merkle(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(insert_values, m)?)?;
    Ok(())
}
