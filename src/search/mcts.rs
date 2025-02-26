use super::{eval::Evaluation, heuristic::random_rollout, node::SpawnNode};
use crate::board::BoardAvx2;
use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Debug,
    time::Instant,
};

#[derive(Debug, Clone, Copy)]
pub struct SearchConstraint {
    pub board: BoardAvx2,
    pub deadline: Instant,
}

#[derive(Default, Clone)]
pub struct Node {
    visits: u32,
    total_reward: u32,
}

impl Node {
    fn evaluate(&self) -> Evaluation {
        let eval = self.total_reward / self.visits;
        debug_assert!(eval <= u16::MAX as u32, "Overflow in evaluate");

        Evaluation::new(eval as u16)
    }

    fn add(&mut self, eval: Evaluation) {
        self.total_reward += eval.as_u32();
        self.visits += 1;
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.evaluate(), self.visits)
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Index(u32);

impl Index {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn idx(self) -> usize {
        self.0 as usize
    }
}

pub struct MonteCarloTreeSearch {
    // TODO: Try storing the swiped boards in the hashmap
    transposition_table: HashMap<u128, Index, fxhash::FxBuildHasher>,
    nodes: Vec<Node>,
    node_stack: Vec<Index>,

    pub exploration_rate: f64,
    pub cache_lookup_counter: u32,
    pub cache_hit_counter: u32,
}

impl MonteCarloTreeSearch {
    pub fn new() -> Self {
        Self {
            transposition_table: HashMap::with_hasher(fxhash::FxBuildHasher::default()),
            nodes: Vec::new(),
            node_stack: Vec::new(),
            exploration_rate: 110.0,
            cache_lookup_counter: 0,
            cache_hit_counter: 0,
        }
    }

    fn select(nodes: &[Node], exploration_rate: f64) -> usize {
        if nodes.len() <= 1 {
            return 0;
        }

        let mut parent_visits = 0;
        let mut max_eval = Evaluation::TERMINAL;
        for (i, node) in nodes.iter().enumerate().rev() {
            if node.visits == 0 {
                return i;
            }

            parent_visits += node.visits;
            max_eval = node.evaluate().max(max_eval);
        }

        let exploration_factor = exploration_rate * (parent_visits as f64).ln();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, node) in nodes.iter().enumerate() {
            let exploit = node.evaluate().as_u16() as f64;
            let explore = (exploration_factor / node.visits as f64).sqrt();
            let score = exploit + explore;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    fn simulate(&mut self, mut board: BoardAvx2) -> Evaluation {
        let mut advantage = 0;
        self.node_stack.clear();
        let mut rng = rand::rng();
        let mut depth = u32::MAX;

        let terminal_eval = loop {
            self.cache_lookup_counter += 1;
            let entry = self.transposition_table.entry(board.as_u128());

            let mut moves = [board; 4];
            let mut num_moves = 0;
            for swiped in board
                .rotations()
                .into_iter()
                .filter_map(|b| b.checked_swipe_right())
            {
                moves[num_moves] = swiped;
                num_moves += 1;
            }

            if num_moves == 0 {
                break Evaluation::TERMINAL;
            }

            if let Entry::Occupied(_) = entry {
                self.cache_hit_counter += 1;
            }

            let index = entry.or_insert_with(|| {
                let index = self.nodes.len();
                self.nodes.extend((0..num_moves).map(|_| Node::default()));
                // Reached leaf, backpropagate
                depth = 0;
                Index::new(index)
            });

            let index = index.idx();
            let children = &self.nodes[index..index + num_moves];
            let move_idx = Self::select(children, self.exploration_rate);
            self.node_stack.push(Index::new(index + move_idx));

            if depth == 0 {
                break random_rollout(&mut rng, moves[move_idx]);
            }

            board = SpawnNode::random_spawned(&mut rng, moves[move_idx]).unwrap();
            advantage += 1;
            depth -= 1;
        };

        terminal_eval + advantage
    }

    fn backpropagate(&mut self, mut advantage: Evaluation) {
        for index in self.node_stack.drain(..) {
            self.nodes[index.idx()].add(advantage);
            advantage -= 1;
        }
    }

    pub fn children(&self, board: BoardAvx2) -> Option<&[Node]> {
        let mut moves = [board; 4];
        let mut num_moves = 0;
        for swiped in board
            .rotations()
            .into_iter()
            .filter_map(|b| b.checked_swipe_right())
        {
            moves[num_moves] = swiped;
            num_moves += 1;
        }

        if num_moves == 0 {
            return Some(&[]);
        }

        let index = self.transposition_table.get(&board.as_u128())?.idx();
        Some(&self.nodes[index..index + num_moves])
    }

    pub fn clear_cache(&mut self) {
        self.transposition_table.clear();
        self.nodes.clear();
        self.node_stack.clear();
        self.cache_lookup_counter = 0;
        self.cache_hit_counter = 0;
    }

    pub fn search(
        &mut self,
        SearchConstraint { board, deadline }: SearchConstraint,
    ) -> ((Evaluation, u16), usize) {
        let mut num_iterations = 0;

        while Instant::now() < deadline {
            let advantage = self.simulate(board);
            self.backpropagate(advantage);
            log::trace!("Advantage: {advantage}");
            num_iterations += 1;
        }

        let mut best = (Evaluation::TERMINAL, 0);

        let mut current_node = match self.transposition_table.get(&board.as_u128()) {
            Some(idx) => idx.idx(),

            // Terminal root
            None => return (best, num_iterations),
        };

        for (move_index, rotation) in board.rotations().into_iter().enumerate() {
            if rotation.checked_swipe_right().is_none() {
                continue;
            }

            let eval = if self.nodes[current_node].visits == 0 {
                Evaluation::new(1)
            } else {
                self.nodes[current_node].evaluate()
            };

            best = (eval, move_index as u16).max(best);
            current_node += 1;
        }

        (best, num_iterations)
    }
}

impl Default for MonteCarloTreeSearch {
    fn default() -> Self {
        Self::new()
    }
}
