use rand::Rng;

use super::{cache::BoardCache, eval::Evaluation, eval::EvaluationState, node::SpawnNode};
use crate::{board::BoardAvx2, search::node::Transition};
use std::{arch::x86_64::__m128i, thread::current};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SearchConstraint {
    pub board: BoardAvx2,
    pub depth: i32,
}

impl SearchConstraint {
    pub const LOGS: [i32; 64] = [
        -100, 0, 100, 158, 200, 232, 258, 280, 300, 316, 332, 345, 358, 370, 380, 390, 400, 408,
        416, 424, 432, 439, 445, 452, 458, 464, 470, 475, 480, 485, 490, 495, 500, 504, 508, 512,
        516, 520, 524, 528, 532, 535, 539, 542, 545, 549, 552, 555, 558, 561, 564, 567, 570, 572,
        575, 578, 580, 583, 585, 588, 590, 593, 595, 597,
    ];

    pub fn new(board: BoardAvx2, depth: i32) -> Self {
        Self { board, depth }
    }

    pub fn sat(&self) -> bool {
        self.depth >= 0
    }

    pub fn tighten(&mut self, factor: u32) {
        debug_assert!(factor > 0);

        //while factor >= 64 {
        //    self.depth -= Self::LOGS[factor as usize % 64];
        //    factor /= 64;
        //}
        //
        //self.depth -= Self::LOGS[factor as usize % 64];
        self.depth -= 1;
    }

    pub fn loosen(&mut self, factor: u32) {
        debug_assert!(factor > 0);

        //while factor >= 64 {
        //    self.depth += Self::LOGS[factor as usize % 64];
        //    factor /= 64;
        //}
        //
        //self.depth += Self::LOGS[factor as usize % 64];
        self.depth += 1;
    }

    pub fn set_depth(&mut self, log_prob: i32) {
        self.depth = log_prob;
    }
}

pub struct MeanMax {
    stack: Vec<u64>,

    /// Number of remaining recursions
    pub iteration_counter: u32,
    pub heuristic_depth: u32,

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
            heuristic_depth: 3,
        }
    }

    pub fn search(
        &mut self,
        SearchConstraint { mut board, depth }: SearchConstraint,
    ) -> (Evaluation, u16) {
        let mut best_move = (Evaluation::WORST, 0);
        self.iteration_counter = 0;

        for move_idx in 0..4 {
            let Some(swiped) = board.checked_swipe_right().and_then(SpawnNode::new) else {
                board = board.rotate_90();
                continue;
            };

            log::trace!("Evaluating move#{move_idx}:\n{swiped:?}");
            let move_eval = self.evaluate_move(swiped, depth, best_move.0.as_fp());
            best_move = (move_eval, move_idx).max(best_move);

            board = board.rotate_90();
        }

        best_move
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

    fn heuristic(&self, mut board: BoardAvx2) -> Evaluation {
        let mut eval = Evaluation::BEST.0;
        let rng = &mut rand::rng();

        for i in 0..self.heuristic_depth {
            let Some(mut node) = SpawnNode::new(board) else {
                eval >>= (self.heuristic_depth - i + 1) as i16;
                break;
            };

            let num_empty = board.num_empty();
            let steps = rng.random_range(0..num_empty * 3);
            if steps >= num_empty * 2 {
                // Spawn a two
                while let Transition::None = node.next_spawn() {}
            }

            for _ in 0..steps % 3 {
                node.next_spawn();
            }

            //match node.random_spawn(rng).rotate_90().checked_swipe_right() {
            match node.current_board().rotate_90().checked_swipe_right() {
                Some(b) => board = b,
                None => break,
            };

            //board = state.current_board();
            //for _ in 0..4 {
            //    if let Some(new_board) = board.checked_swipe_right() {
            //        eval += 10;
            //        board = new_board;
            //        continue 'outer;
            //    };
            //
            //    board = board.rotate_90();
            //}
            //
            //break;
        }

        let num_empty = board.num_empty() as i16;

        eval += (1 << num_empty.min(5)) - 32;
        eval *= 1;
        //let b = 8 * num_empty;

        //Evaluation(a + b)
        Evaluation(eval)
    }

    #[inline(never)]
    fn evaluate_move(
        &mut self,
        node: SpawnNode,
        mut depth: i32,
        mut lower_bound: i32,
    ) -> Evaluation {
        #[derive(Debug)]
        enum Action {
            Evaluate(SpawnNode),
            Expand(SpawnNode, EvaluationState),
            Propagate(Evaluation),
            Branch(EvaluationState),
        }

        let mut action = Action::Evaluate(node);

        loop {
            self.iteration_counter += 1;
            log::trace!("Action: {action:?}");
            log::trace!("Depth: {depth:2}, LowerB: {lower_bound}");

            action = match action {
                Action::Evaluate(node) => {
                    if let Some(eval) = self.eval_cache.get(node.inner(), depth) {
                        Action::Propagate(*eval)
                    } else if self
                        .prune_cache
                        .get(node.inner(), depth)
                        .is_some_and(|ub| ub.as_fp() <= lower_bound)
                    {
                        Action::Propagate(Evaluation(lower_bound.try_into().unwrap()))
                    } else if depth <= 0 {
                        Action::Propagate(self.heuristic(node.inner()))
                    } else {
                        depth -= 1;
                        Action::Expand(node, EvaluationState::new())
                    }
                }

                Action::Expand(node, mut eval_state) => {
                    let rot0 = node.current_board();
                    let rot1 = rot0.rotate_90();
                    let rot2 = rot1.rotate_90();
                    let rot3 = rot2.rotate_90();

                    self.push_node(node);

                    eval_state.reset_moves();
                    let mut nodes = [rot0, rot1, rot2, rot3]
                        .into_iter()
                        .filter_map(|b| b.checked_swipe_right())
                        .filter_map(SpawnNode::new)
                        .inspect(|_| eval_state.add_move());

                    if let Some(node) = nodes.next() {
                        nodes.for_each(|b| self.push_node(b));
                        self.push_state(eval_state);
                        Action::Evaluate(node)
                    } else {
                        eval_state.push_spawn_eval(Evaluation::TERM);
                        Action::Branch(eval_state)
                    }
                }

                Action::Propagate(eval) => {
                    // Reached root
                    let Some(mut eval_state) = self.stack.pop().map(EvaluationState::from) else {
                        return eval;
                    };

                    let mut is_last_move = eval_state.push_move_eval(eval);
                    lower_bound = eval_state.best_move_eval.as_fp();

                    if lower_bound >= Evaluation::BEST.as_fp() {
                        // Prune action
                        //if !is_last_move { self.pop_node();
                        //}
                        //
                        //while !eval_state.push_move_eval(eval) {
                        //    self.pop_node();
                        //}

                        while eval_state.remaining_moves > 0 {
                            self.pop_node();
                            eval_state.push_move_eval(eval);
                        }

                        is_last_move = true;
                    }

                    if is_last_move {
                        // This was the last move, spawn
                        Action::Branch(eval_state)
                    } else {
                        // We have other moves to evaluate
                        let node = self.pop_node();

                        // Evaluate next move
                        self.push_state(eval_state);
                        Action::Evaluate(node)
                    }
                }

                Action::Branch(mut state) => {
                    let mut node = self.pop_node();
                    let board = node.inner();

                    // Recompute lower_bound:
                    let max_denominator = board.num_empty() as i32 * 3;
                    let max_eval = Evaluation::BEST.as_fp();

                    let transition = node.next_spawn();
                    if let Transition::Switch = transition {
                        state.switch();
                    }

                    if let Transition::Done = transition {
                        let eval = state.evaluate();
                        self.eval_cache.insert(board, depth, eval);

                        depth += 1;
                        Action::Propagate(eval)
                    } else {
                        let lb_origin = lower_bound;
                        let max_loss = (max_eval - lower_bound) * max_denominator;
                        let gain = state.denominator as i32 * max_eval - state.numerator as i32;
                        lower_bound = max_eval - max_loss + gain;
                        if lower_bound > max_eval {
                            // Prune
                            depth += 1;

                            self.prune_cache
                                .insert(board, depth, Evaluation(lb_origin as i16));

                            Action::Propagate(state.evaluate())
                        } else {
                            Action::Expand(node, state)
                        }
                    }
                }
            };
        }
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
        let (eval, move_idx) = mean_max.search(SearchConstraint { board, depth: 2 });

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(eval, Evaluation(150));
    }

    #[test]
    fn test_mean_max_1() {
        let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let (eval, move_idx) = mean_max.search(SearchConstraint { board, depth: 2 });

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(eval, Evaluation(511));
    }

    #[test]
    fn test_mean_max_2() {
        let cells = [[1, 2, 3, 4], [5, 1, 3, 2], [1, 2, 3, 0], [0, 0, 0, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let (eval, move_idx) = mean_max.search(SearchConstraint { board, depth: 3 });

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(eval, Evaluation(511));
    }
}
