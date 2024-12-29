// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use blake3::{Hash as Blake3Hash, Hasher};
use dusk_merkle::{Aggregate, Tree};
use serde::Serialize;
use hex;

const EMPTY_HASH: Item = Item([0; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
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

const H: usize = 16;
const A: usize = 2;

fn main() {
    let mut tree = Tree::<Item, H, A>::new();

    // fill the first 1000 Items in the tree
    for pos in 0..1000u64 {
        let hash_bytes = pos.to_be_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&hash_bytes);
        let hash: Item = hasher.finalize().into();

        tree.insert(pos, hash);
    }

    // check that there is a leaf at pos 42 and remove it
    let pos = 42;
    assert!(tree.contains(pos));
    let leaf = tree.remove(42).expect("There is a leaf at this position");

    // insert the leaf back into the tree
    tree.insert(pos, leaf);

    // create opening from position 42 and verify it
    let opening = tree.opening(42).expect("There is a leaf at this position");
    assert!(opening.verify(leaf));

    println!("Tree: {:?}", tree.root().0);
    println!("Root: 0x{}", hex::encode(tree.root().0));
}
