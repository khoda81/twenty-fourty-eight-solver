use std::collections::{HashMap, hash_map::Entry};

use crate::board::BoardAvx2;

use super::search_state::Evaluation;

#[derive(Clone)]
struct CacheEntry {
    depth: u32,
    eval: Evaluation,
}

pub struct EvaluationCache {
    cache: HashMap<u128, CacheEntry>,
    hit_counter: u32,
    lookup_counter: u32,
}

impl EvaluationCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hit_counter: 0,
            lookup_counter: 0,
        }
    }

    pub fn get(&mut self, board: BoardAvx2, depth: u32) -> Option<Evaluation> {
        self.lookup_counter += 1;

        match self.cache.get(&board.into_u128()) {
            Some(entry) if entry.depth >= depth => {
                self.hit_counter += 1;
                Some(entry.eval.clone())
            }
            _ => None,
        }
    }

    pub fn insert(&mut self, board: BoardAvx2, depth: u32, eval: Evaluation) {
        let cache_entry = CacheEntry { depth, eval };

        self.cache
            .entry(board.into_u128())
            .and_modify(|entry| {
                if entry.depth <= depth {
                    *entry = cache_entry.clone();
                }
            })
            .or_insert(cache_entry);
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
}

impl Default for EvaluationCache {
    fn default() -> Self {
        Self::new()
    }
}
