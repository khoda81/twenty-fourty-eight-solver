use crate::board::BoardAvx2;
use std::collections::HashMap;

#[derive(Clone)]
struct CacheEntry<T> {
    depth: i32,
    data: T,
}

pub struct BoardCache<T> {
    cache: HashMap<u128, CacheEntry<T>>,
    hit_counter: u32,
    lookup_counter: u32,
}

impl<T> BoardCache<T> {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hit_counter: 0,
            lookup_counter: 0,
        }
    }

    pub fn get(&mut self, board: BoardAvx2, depth: i32) -> Option<&T> {
        self.lookup_counter += 1;

        match self.cache.get(&board.into_u128()) {
            Some(entry) if entry.depth >= depth => {
                self.hit_counter += 1;
                Some(&entry.data)
            }
            _ => None,
        }
    }

    pub fn insert(&mut self, board: BoardAvx2, depth: i32, eval: T) {
        let cache_entry = CacheEntry { depth, data: eval };
        use std::collections::hash_map::Entry;

        match self.cache.entry(board.into_u128()) {
            Entry::Occupied(mut occupied_entry) if occupied_entry.get().depth <= depth => {
                occupied_entry.insert(cache_entry);
            }

            Entry::Occupied(_) => { /* Do nothing */ }

            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(cache_entry);
            }
        };
    }

    pub fn hit_rate(&self) -> f64 {
        self.hit_counter as f64 / self.lookup_counter as f64
    }

    pub fn hit_counter(&self) -> u32 {
        self.hit_counter
    }

    pub fn lookup_counter(&self) -> u32 {
        self.lookup_counter
    }

    pub(crate) fn clear(&mut self) {
        self.cache.clear();
        self.hit_counter = 0;
        self.lookup_counter = 0;
    }
}

impl<T> Default for BoardCache<T> {
    fn default() -> Self {
        Self::new()
    }
}
