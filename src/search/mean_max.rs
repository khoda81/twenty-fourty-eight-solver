use super::{
    cache::EvaluationCache,
    search_state::{Evaluation, EvaluationState, SpawnIter},
};
use crate::{board::BoardAvx2, search::search_state::Transition};
use std::arch::x86_64::__m128i;

pub struct MeanMax {
    stack: Vec<u64>,
    /// Number of remaining recursions
    pub depth: u32,
    pub iteration_counter: u32,

    cache: EvaluationCache,
}

impl MeanMax {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            depth: 0,
            cache: EvaluationCache::new(),
            iteration_counter: 0,
        }
    }

    pub fn best_move(&mut self, mut board: BoardAvx2, depth: u32) -> (u16, Evaluation) {
        let mut best_move = 0;
        let mut best_eval = Evaluation::MIN;

        for move_idx in 0..4 {
            let Some(swiped) = board.checked_swipe_right() else {
                board = board.rotate_90();
                continue;
            };

            eprintln!("Evaluating move#{move_idx}:\n{swiped:?}");
            self.depth = depth;
            let swipe_eval = self.evaluate_move(swiped);
            debug_assert_eq!(self.depth, depth);

            if swipe_eval >= best_eval {
                best_move = move_idx;
                best_eval = swipe_eval;
            }

            board = board.rotate_90();
        }

        (best_move, best_eval)
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

    fn heuristic(&self, board: BoardAvx2) -> Evaluation {
        let num_empty = board.num_empty();
        Evaluation(10 * (1 << num_empty))
    }

    #[inline(never)]
    fn evaluate_move(&mut self, board: BoardAvx2) -> Evaluation {
        enum State {
            EvaluateMove(BoardAvx2),
            ExpandMoves(SpawnIter, EvaluationState),
            UpdateMoveEval(Evaluation),
            NextSpawn(EvaluationState),
        }

        let mut state = State::EvaluateMove(board);

        loop {
            self.iteration_counter += 1;
            state = match state {
                State::EvaluateMove(board) => {
                    if let Some(eval) = self.cache.get(board, self.depth) {
                        State::UpdateMoveEval(eval.clone())
                    } else if self.depth == 0 {
                        State::UpdateMoveEval(self.heuristic(board))
                    } else if let Some(iter) = SpawnIter::new(board) {
                        self.depth -= 1;
                        State::ExpandMoves(iter, EvaluationState::new())
                    } else {
                        unreachable!()
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
                        .map(BoardAvx2::checked_swipe_right)
                        .into_iter()
                        .flatten()
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
                    let Some(mut state) = self.stack.pop().map(EvaluationState::from) else {
                        return eval;
                    };

                    if state.push_move_eval(eval) {
                        // This was the last move, spawn
                        State::NextSpawn(state)
                    } else {
                        // Evaluate next move
                        let board = BoardAvx2(self.pop_xmm());
                        self.push_state(state);
                        State::EvaluateMove(board)
                    }
                }

                State::NextSpawn(state) => {
                    let mut iter = SpawnIter(self.pop_xmm());
                    let board = iter.board();
                    match iter.next_spawn() {
                        Transition::None => State::ExpandMoves(iter, state),
                        Transition::Switch => State::ExpandMoves(iter, state.switch()),
                        Transition::Done => {
                            self.depth += 1;
                            let eval = state.evaluate();
                            self.cache.insert(board, self.depth, eval.clone());

                            State::UpdateMoveEval(eval)
                        }
                    }
                }
            };
        }
    }

    pub fn cache(&self) -> &EvaluationCache {
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
    use crate::{board::BoardAvx2, search::search_state::Evaluation};

    #[test]
    fn test_mean_max() {
        let cells = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let (move_idx, eval) = mean_max.best_move(board, 9);

        eprintln!(
            "Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {}",
            eval.0
        );
        eprintln!("Iterations: {}", mean_max.iteration_counter);

        assert_eq!(move_idx, 2);
        assert_eq!(eval, Evaluation(-1161));
        assert_eq!(mean_max.iteration_counter, 3450052);
    }

    #[test]
    fn test_mean_max_1() {
        let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let board = BoardAvx2::from_array(cells).unwrap();

        let mut mean_max = MeanMax::new();
        let (move_idx, eval) = mean_max.best_move(board, 3);

        assert_eq!(move_idx, 3);
        assert_eq!(eval, Evaluation(0));
        assert_eq!(mean_max.iteration_counter, 12051540);
    }
}
