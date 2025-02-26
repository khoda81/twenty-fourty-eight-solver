use thiserror::Error;

use super::{cache::BoardCache, eval::Evaluation, eval::EvaluationState, node::SpawnNode};
use crate::{board::BoardAvx2, search::node::Transition};
use std::{
    arch::x86_64::__m128i,
    time::{Duration, Instant},
};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SearchConstraint {
    pub board: BoardAvx2,
    pub depth: Option<i32>,
    pub deadline: Option<Instant>,
}

impl SearchConstraint {
    pub const LOGS: [i32; 64] = [
        -100, 0, 100, 158, 200, 232, 258, 280, 300, 316, 332, 345, 358, 370, 380, 390, 400, 408,
        416, 424, 432, 439, 445, 452, 458, 464, 470, 475, 480, 485, 490, 495, 500, 504, 508, 512,
        516, 520, 524, 528, 532, 535, 539, 542, 545, 549, 552, 555, 558, 561, 564, 567, 570, 572,
        575, 578, 580, 583, 585, 588, 590, 593, 595, 597,
    ];

    pub fn new(board: BoardAvx2) -> Self {
        Self {
            board,
            depth: None,
            deadline: None,
        }
    }

    pub fn depth(self, depth: i32) -> Self {
        Self {
            depth: Some(depth),
            ..self
        }
    }

    pub fn deadline(self, deadline: Instant) -> Self {
        Self {
            deadline: Some(deadline),
            ..self
        }
    }

    pub fn deadline_from_now(self, deadline: Duration) -> Self {
        Self {
            deadline: Some(Instant::now() + deadline),
            ..self
        }
    }

    //pub fn tighten(&mut self, factor: u32) {
    //    debug_assert!(factor > 0);
    //
    //    //while factor >= 64 {
    //    //    self.depth -= Self::LOGS[factor as usize % 64];
    //    //    factor /= 64;
    //    //}
    //    //
    //    //self.depth -= Self::LOGS[factor as usize % 64];
    //    self.depth -= 1;
    //}
    //
    //pub fn loosen(&mut self, factor: u32) {
    //    debug_assert!(factor > 0);
    //
    //    //while factor >= 64 {
    //    //    self.depth += Self::LOGS[factor as usize % 64];
    //    //    factor /= 64;
    //    //}
    //    //
    //    //self.depth += Self::LOGS[factor as usize % 64];
    //    self.depth += 1;
    //}
}

#[derive(Debug, Error)]
#[error("Deadline reached before the search was complete")]
pub struct SearchError;

pub struct MeanMax {
    stack: Vec<u64>,

    /// Number of remaining recursions
    pub iteration_counter: u32,

    eval_cache: BoardCache<Evaluation>,
    prune_cache: BoardCache<Evaluation>,
}

impl MeanMax {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            eval_cache: BoardCache::new(),
            prune_cache: BoardCache::new(),
            iteration_counter: 0,
        }
    }

    pub fn search_flexible(&mut self, mut constraint: SearchConstraint) -> (Evaluation, u16) {
        self.iteration_counter = 0;
        let deadline = constraint.deadline.take();
        let mut result = self
            .search_fixed(constraint)
            .expect("search should not fail without deadline");

        constraint.deadline = deadline;
        constraint.depth = constraint.depth.and_then(|d| d.checked_add(1));

        while let Ok(best_move) = self.search_fixed(constraint) {
            result = best_move;

            constraint.depth = constraint.depth.and_then(|d| d.checked_add(1));
        }

        result
    }

    pub fn search_dynamic(
        &mut self,
        mut constraint: SearchConstraint,
    ) -> Result<(Evaluation, u16), SearchError> {
        self.iteration_counter = 0;
        let mut result = Err(SearchError);

        loop {
            match self.search_fixed(constraint) {
                Ok(best_move) => result = Ok(best_move),
                Err(search_error) => break Err(search_error).or(result),
            }

            constraint.depth = constraint.depth.and_then(|d| d.checked_add(1));
        }
    }

    pub fn search_fixed(
        &mut self,
        SearchConstraint {
            mut board,
            depth,
            deadline,
        }: SearchConstraint,
    ) -> Result<(Evaluation, u16), SearchError> {
        let search_duration = deadline.map(|d| d.duration_since(Instant::now()));
        let depth = depth.unwrap_or(i32::MAX);

        if search_duration.is_some_and(|d| d.is_zero()) {
            return Err(SearchError);
        }

        if let Some(duration) = search_duration {
            log::debug!("Searching for {:?} (depth={depth:?})", duration);
        } else {
            log::debug!("Searching for ever (depth={depth:?})");
        }

        let mut best_move = (Evaluation::WORST, 0);

        for move_idx in 0..4 {
            let Some(swiped) = board.checked_swipe_right().and_then(SpawnNode::new) else {
                board = board.rotate_90();
                continue;
            };

            log::trace!("Evaluating move#{move_idx}:\n{swiped:?}");
            let move_eval = self.evaluate_move(swiped, depth, best_move.0, deadline)?;

            best_move = (move_eval, move_idx).max(best_move);
            board = board.rotate_90();
        }

        log::debug!("Result: {best_move:?}");

        Ok(best_move)
    }

    pub fn clear_cache(&mut self) {
        self.eval_cache.clear();
        self.prune_cache.clear();
    }

    fn push_node(&mut self, node: SpawnNode) {
        let words: [u64; 2] =
            unsafe { std::mem::transmute::<__m128i, [u64; 2]>(node.into_inner()) };

        self.stack.extend(words);
    }

    fn pop_node(&mut self) -> SpawnNode {
        debug_assert!(self.stack.len() >= 2);

        unsafe {
            let a = self.stack.pop().unwrap_unchecked();
            let b = self.stack.pop().unwrap_unchecked();
            let _src = [b, a];
            SpawnNode(std::mem::transmute::<[u64; 2], __m128i>(_src))
        }
    }

    fn push_state(&mut self, eval: EvaluationState) {
        self.stack.push(eval.into());
    }

    #[inline(never)]
    fn evaluate_move(
        &mut self,
        mut node: SpawnNode,
        mut depth: i32,
        mut lower_bound: Evaluation,
        deadline: Option<Instant>,
    ) -> Result<Evaluation, SearchError> {
        #[derive(Debug)]
        enum Action {
            Evaluate,
            Expand,
            Propagate(Evaluation),
            Branch,
        }

        let mut action = Action::Evaluate;
        let mut state = EvaluationState::new(lower_bound, node.board());
        debug_assert!(self.stack.is_empty(), "stack should be empty before search");

        while deadline.is_none_or(|i| i.elapsed().is_zero()) {
            self.iteration_counter += 1;
            log::trace!("Action: {action:?}");
            log::trace!("Depth: {depth:2}, LowerB: {lower_bound}");

            action = match action {
                Action::Evaluate => {
                    if let Some(eval) = self.eval_cache.get(node.board(), depth) {
                        Action::Propagate(*eval)
                    } else if depth <= 0 {
                        Action::Propagate(super::heuristic::heuristic(node.board()))
                    } else {
                        depth -= 1;
                        state = EvaluationState::new(lower_bound, node.board());
                        Action::Expand
                    }
                }

                Action::Expand => {
                    lower_bound = state.required_lower_bound();

                    let mut nodes = node
                        .current_branch()
                        .rotations()
                        .into_iter()
                        .filter_map(BoardAvx2::checked_swipe_right)
                        .filter_map(SpawnNode::new)
                        .inspect(|_| state.add_move());

                    if let Some(next_node) = nodes.next() {
                        self.push_node(node);
                        nodes.for_each(|b| self.push_node(b));

                        node = next_node;
                        self.push_state(state.clone());
                        Action::Evaluate
                    } else {
                        state.push_spawn_eval(Evaluation::TERMINAL);
                        Action::Branch
                    }
                }

                Action::Propagate(eval) => {
                    // Reached root
                    state = match self.stack.pop().map(EvaluationState::from) {
                        Some(state) => state,
                        None => return Ok(eval),
                    };

                    if eval >= Evaluation::BEST {
                        // Reached max eval, prune action
                        state.push_move_eval(eval);
                        while state.remaining_moves > 0 {
                            self.pop_node();
                            state.push_move_eval(eval);
                        }

                        node = self.pop_node();
                        Action::Branch
                    } else if state.push_move_eval(eval) {
                        // This was the last move, spawn
                        node = self.pop_node();
                        Action::Branch
                    } else {
                        // We have other moves to evaluate
                        lower_bound = state.best_move_eval;

                        node = self.pop_node();
                        self.push_state(state.clone());
                        Action::Evaluate
                    }
                }

                Action::Branch => {
                    let board = node.board();

                    let transition = node.next_spawn();
                    if let Transition::Switch = transition {
                        state.switch();
                    }

                    if let Transition::Done = transition {
                        let eval = state.evaluate();
                        self.eval_cache.insert(board, depth, eval);

                        depth += 1;
                        Action::Propagate(eval)
                    } else if state.prunable() {
                        // Prune
                        let eval = state.evaluate();

                        //let upper_bound = state.upper_bound(board);
                        //self.prune_cache.insert(board, depth, upper_bound);

                        depth += 1;
                        Action::Propagate(eval)
                    } else {
                        state.next_branch();
                        Action::Expand
                    }
                }
            };
        }

        self.stack.clear();
        Err(SearchError)
    }

    pub fn eval_cache(&self) -> &BoardCache<Evaluation> {
        &self.eval_cache
    }

    pub fn prune_cache(&self) -> &BoardCache<Evaluation> {
        &self.prune_cache
    }
}

impl Default for MeanMax {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use super::MeanMax;
    use crate::{
        board::BoardAvx2,
        search::{eval::Evaluation, mean_max::SearchConstraint},
    };

    #[test]
    fn test_mean_max() {
        let cells = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let constraint = SearchConstraint::new(board)
            .depth(2)
            .deadline_from_now(Duration::from_secs(1));

        let (eval, move_idx) = mean_max.search_dynamic(constraint).unwrap();

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(eval, Evaluation::new(172));
    }

    #[test]
    fn test_mean_max_1() {
        let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let constraint = SearchConstraint::new(board)
            .depth(2)
            .deadline_from_now(Duration::from_secs(1));

        let (eval, move_idx) = mean_max.search_fixed(constraint).unwrap();

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(eval, Evaluation::new(1455));
    }

    #[test]
    fn test_mean_max_2() {
        let cells = [[1, 2, 3, 4], [5, 1, 3, 2], [1, 2, 3, 0], [0, 0, 0, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let constraint = SearchConstraint::new(board).depth(3);

        let (eval, move_idx) = mean_max.search_fixed(constraint).unwrap();

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(eval, Evaluation::new(848));
    }
}
