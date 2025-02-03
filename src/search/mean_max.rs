use super::{
    cache::BoardCache, modname::Evaluation, modname::EvaluationState, search_state::SpawnIter,
};
use crate::{board::BoardAvx2, search::search_state::Transition};
use std::arch::x86_64::__m128i;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SearchConstraint {
    pub depth: i32,
}

impl SearchConstraint {
    pub const LOGS: [i32; 64] = [
        -100, 0, 100, 158, 200, 232, 258, 280, 300, 316, 332, 345, 358, 370, 380, 390, 400, 408,
        416, 424, 432, 439, 445, 452, 458, 464, 470, 475, 480, 485, 490, 495, 500, 504, 508, 512,
        516, 520, 524, 528, 532, 535, 539, 542, 545, 549, 552, 555, 558, 561, 564, 567, 570, 572,
        575, 578, 580, 583, 585, 588, 590, 593, 595, 597,
    ];

    pub fn new(log_prob: i32) -> Self {
        Self { depth: log_prob }
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
    pub search_constraint: SearchConstraint,
    pub iteration_counter: u32,

    cache: BoardCache<Evaluation>,
}

#[derive(Debug)]
enum State {
    EvaluateMove(BoardAvx2),
    ExpandMoves(SpawnIter, EvaluationState),
    UpdateMoveEval(Evaluation),
    NextSpawn(EvaluationState),
}

impl MeanMax {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            search_constraint: SearchConstraint::new(0),
            cache: BoardCache::new(),
            iteration_counter: 0,
        }
    }

    pub fn best_move(&mut self, mut board: BoardAvx2) -> (Evaluation, u16) {
        let mut best_move = (Evaluation::MIN, 0);

        for move_idx in 0..4 {
            let Some(swiped) = board.checked_swipe_right() else {
                board = board.rotate_90();
                continue;
            };

            log::trace!("Evaluating move#{move_idx}:\n{swiped}");

            let constraint = self.search_constraint.clone();
            best_move = (self.evaluate_move(swiped), move_idx).max(best_move);
            debug_assert_eq!(self.search_constraint, constraint);

            board = board.rotate_90();
        }

        best_move
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear()
    }

    fn pop_xmm(&mut self) -> __m128i {
        debug_assert!(self.stack.len() >= 2);

        unsafe {
            let a = self.stack.pop().unwrap_unchecked();
            let b = self.stack.pop().unwrap_unchecked();
            let _src = [b, a];
            std::mem::transmute(_src)
        }
    }

    fn push_xmm(&mut self, board: __m128i) {
        let words: [u64; 2] = unsafe { std::mem::transmute(board) };
        self.stack.extend(words);
    }

    fn push_state(&mut self, eval: EvaluationState) {
        self.stack.push(eval.into());
    }

    fn heuristic(&self, mut board: BoardAvx2) -> Evaluation {
        let mut eval = 0;

        //const N: i16 = 8;
        //for _ in 0..N {
        //    let Some(state) = SpawnIter::new(board) else {
        //        break;
        //    };
        //
        //    match state.current_board().rotate_90().checked_swipe_right() {
        //        Some(b) => board = b,
        //        None => break,
        //    };
        //
        //    eval += 10;
        //
        //    //board = state.current_board();
        //    //for _ in 0..4 {
        //    //    if let Some(new_board) = board.checked_swipe_right() {
        //    //        eval += 10;
        //    //        board = new_board;
        //    //        continue 'outer;
        //    //    };
        //    //
        //    //    board = board.rotate_90();
        //    //}
        //    //
        //    //break;
        //}

        let num_empty = board.num_empty() as i16;

        eval += 1 << num_empty.min(5);
        eval *= 5;
        //let b = 8 * num_empty;

        //Evaluation(a + b)
        Evaluation(eval)
    }

    #[inline(never)]
    fn evaluate_move(&mut self, board: BoardAvx2) -> Evaluation {
        let mut state = State::EvaluateMove(board);

        loop {
            self.iteration_counter += 1;
            log::trace!("State: {state:?}, constraint: {:?}", self.search_constraint);

            state = self.handle_state(state);
            if let State::UpdateMoveEval(value) = &state {
                if self.stack.is_empty() {
                    return value.clone();
                }
            }
        }
    }

    #[inline(always)]
    fn handle_state(&mut self, state: State) -> State {
        match state {
            State::EvaluateMove(board) => {
                if let Some(eval) = self.cache.get(board, self.search_constraint.depth) {
                    State::UpdateMoveEval(eval.clone())
                } else if !self.search_constraint.sat() {
                    State::UpdateMoveEval(self.heuristic(board))
                } else if let Some(iter) = SpawnIter::new(board) {
                    self.search_constraint.tighten(board.num_empty());
                    State::ExpandMoves(iter, EvaluationState::new())
                } else {
                    unreachable!("moves that need to be evaluated will always have an empty cell")
                }
            }

            State::ExpandMoves(iter, mut eval_state) => {
                let rot0 = iter.current_board();
                let rot1 = rot0.rotate_90();
                let rot2 = rot1.rotate_90();
                let rot3 = rot2.rotate_90();

                self.push_xmm(iter.into_inner());

                eval_state.reset_moves();
                let mut filtered_boards = [rot0, rot1, rot2, rot3]
                    .into_iter()
                    .filter_map(BoardAvx2::checked_swipe_right)
                    .inspect(|_| eval_state.add_move());

                if let Some(next_board) = filtered_boards.next() {
                    filtered_boards.for_each(|b| self.push_xmm(b.into_inner()));
                    self.push_state(eval_state);
                    State::EvaluateMove(next_board)
                } else {
                    State::NextSpawn(eval_state.push_spawn_eval(Evaluation::TERM))
                }
            }

            State::UpdateMoveEval(eval) => {
                // Reached root
                let mut eval_state = self.stack.pop().map(EvaluationState::from).unwrap();

                if eval_state.push_move_eval(eval) {
                    // This was the last move, spawn
                    return State::NextSpawn(eval_state);
                }

                // Evaluate next move
                let board = BoardAvx2(self.pop_xmm());
                self.push_state(eval_state);
                State::EvaluateMove(board)
            }

            State::NextSpawn(state) => {
                let mut iter = SpawnIter(self.pop_xmm());
                let board = iter.board();
                match iter.next_spawn() {
                    Transition::None => State::ExpandMoves(iter, state),
                    Transition::Switch => State::ExpandMoves(iter, state.switch()),
                    Transition::Done => {
                        self.search_constraint.loosen(board.num_empty());
                        let eval = state.evaluate();
                        self.cache
                            .insert(board, self.search_constraint.depth, eval.clone());

                        State::UpdateMoveEval(eval)
                    }
                }
            }
        }
    }

    fn expand_rotations(&mut self, iter: SpawnIter, mut eval_state: EvaluationState) -> State {
        let rot0 = iter.current_board();
        let rot1 = rot0.rotate_90();
        let rot2 = rot1.rotate_90();
        let rot3 = rot2.rotate_90();

        self.push_xmm(iter.into_inner());

        eval_state.reset_moves();
        eval_state.add_move();
        eval_state.add_move();
        eval_state.add_move();
        eval_state.add_move();

        self.push_xmm(rot1.into_inner());
        self.push_xmm(rot2.into_inner());
        self.push_xmm(rot3.into_inner());
        self.push_state(eval_state);

        State::EvaluateMove(rot0)
    }

    pub fn cache(&self) -> &BoardCache<Evaluation> {
        &self.cache
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
    use crate::{board::BoardAvx2, search::modname::Evaluation};

    #[test]
    fn test_mean_max() {
        let cells = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        mean_max.search_constraint.set_depth(6);
        let (eval, move_idx) = mean_max.best_move(board);

        log::info!("Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {eval}");
        log::info!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(move_idx, 2);
        assert_eq!(eval, Evaluation(-1146));
        assert_eq!(mean_max.iteration_counter, 7944);
    }

    #[test]
    fn test_mean_max_1() {
        let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        mean_max.search_constraint.set_depth(3);
        let (eval, move_idx) = mean_max.best_move(board);

        assert_eq!(move_idx, 3);
        assert_eq!(eval, Evaluation(560));
        assert_eq!(mean_max.iteration_counter, 165376);
    }
}
