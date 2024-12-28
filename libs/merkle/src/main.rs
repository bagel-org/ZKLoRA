use pyo3::prelude::*;

use dusk_merkle::{Tree, Aggregate};

#[derive(Debug, Clone, Copy, PartialEq)]
struct U8(u8);

impl From<u8> for U8 {
    fn from(n: u8) -> Self {
        Self(n)
    }
}

const EMPTY_ITEM: U8 = U8(0);

impl Aggregate<A> for U8 {
    const EMPTY_SUBTREE: U8 = EMPTY_ITEM;

    fn aggregate(items: [&Self; A]) -> Self
    {
        items.into_iter().fold(U8(0), |acc, c| U8(acc.0 + c.0))
    }
}

// Set the height and arity of the tree. 
const H: usize = 3;
const A: usize = 2;

fn main() {
    let mut tree = Tree::<U8, H, A>::new();

    // No elements have been inserted so the root is the empty subtree.
    assert_eq!(*tree.root(), U8::EMPTY_SUBTREE);

    tree.insert(4, 21);
    tree.insert(7, 21);

    // After elements have been inserted, the root will be modified.
    assert_eq!(*tree.root(), U8(42));
}