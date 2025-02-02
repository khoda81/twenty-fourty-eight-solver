use std::arch::x86_64::*;

use crate::board::BoardAvx2;

#[derive(Debug)]
pub struct SpawnIter(pub(crate) __m128i);

#[derive(Debug)]
pub enum Transition {
    Done,
    Switch,
    None,
}

impl SpawnIter {
    pub fn from_board(board: BoardAvx2) -> Option<Self> {
        unsafe {
            let inner = board.into_inner();
            let msb = _mm_movemask_epi8(inner) as u16;
            crate::debug_println!("msb: {msb:016b}");

            if msb == u16::MAX {
                return None;
            }

            let next_spawn = msb.trailing_ones() as u8;
            crate::debug_println!("next: {next_spawn:?}");

            let mut target = [0; 16];
            target[next_spawn as usize] = 0b1110_0001_u8;

            crate::debug_println!("target: {target:?}");

            let target = _mm_loadu_si128(target.as_ptr().cast());
            let inner = _mm_blendv_epi8(inner, target, target);

            Some(Self(inner))
        }
    }

    pub fn next_spawn(&mut self) -> Transition {
        use std::mem::transmute as tb;

        let Self(mut inner) = *self;

        let shifted = unsafe { _mm_slli_epi16::<1>(inner) };
        let msb = unsafe { _mm_movemask_epi8(inner) as u16 };
        crate::debug_println!("msb:   {msb:016b}");
        let state = unsafe { _mm_movemask_epi8(shifted) as u16 };
        crate::debug_println!("state: {state:016b}");
        let mut result = Transition::None;

        let mut filled_mask = msb | state.wrapping_sub(1);
        if filled_mask == u16::MAX {
            // Create a mask to extract the shifted value at previous_spawn position
            inner = unsafe { _mm_blendv_epi8(inner, shifted, shifted) };
            filled_mask = msb;
            result = Transition::Switch;
        }

        let next_spawn_idx = (filled_mask & !(1 << 15)).trailing_ones() as usize;
        let previous_spawn = (state | (1 << 15)).trailing_zeros() as usize;

        unsafe {
            let mut cells: [u8; 16] = tb(inner);
            cells.swap_unchecked(next_spawn_idx, previous_spawn);

            self.0 = _mm_loadu_si128(cells.as_ptr().cast());

            let state_mask = _mm_set1_epi8(0b01000000);
            let state_m = _mm_and_si128(self.0, state_mask);

            if _mm_movemask_epi8(_mm_cmpeq_epi8(state_m, _mm_setzero_si128())) == 0xFFFF {
                Transition::Done
            } else {
                result
            }
        }
    }

    pub fn current_board(&self) -> BoardAvx2 {
        unsafe {
            let mask = _mm_set1_epi8(0b10011111_u8 as i8);
            BoardAvx2(_mm_and_si128(self.0, mask))
        }
    }

    pub fn into_inner(self) -> __m128i {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Evaluation(pub i16);

impl Evaluation {
    pub const TERM: Self = Evaluation(-0x00FF);
    pub const MIN: Self = Evaluation(-0x00FF);
}

#[repr(C)]
#[derive(Debug, Clone)]
pub(super) struct EvaluationState {
    numerator: i16,
    denominator: u16,
    maximum_eval: Evaluation,
    remaining_moves: u16,
}

impl EvaluationState {
    pub fn new() -> EvaluationState {
        EvaluationState {
            numerator: 0,
            denominator: 0,
            maximum_eval: Evaluation::MIN,
            remaining_moves: 0,
        }
    }

    pub fn push_move_eval(&mut self, eval: Evaluation) -> bool {
        debug_assert!(self.remaining_moves > 0);
        self.maximum_eval = self.maximum_eval.clone().max(eval);
        self.remaining_moves -= 1;

        let done = self.remaining_moves == 0;
        if done {
            *self = self.clone().push_spawn_eval(self.maximum_eval.clone())
        }

        done
    }

    pub fn push_spawn_eval(mut self, value: Evaluation) -> Self {
        self.numerator += value.0;
        self.denominator += 1;
        self
    }

    pub fn evaluate(self) -> Evaluation {
        debug_assert_eq!(self.remaining_moves, 0, "we have not tried all the moves");
        debug_assert_eq!(self.denominator % 3, 0, "denominator not divisible by 3");
        debug_assert!(self.denominator <= 3 * 15, "denominator not divisible by 3");
        Evaluation(self.numerator / self.denominator as i16)
    }

    pub fn switch(mut self) -> Self {
        debug_assert!(self.denominator <= 15, "denominator not divisible by 3");
        self.numerator *= 2;
        self.denominator *= 2;
        self
    }

    pub fn reset_moves(&mut self) {
        self.remaining_moves = 0;
        self.maximum_eval = Evaluation::MIN;
    }

    pub fn add_move(&mut self) {
        self.remaining_moves += 1;
    }
}

impl From<u64> for EvaluationState {
    fn from(value: u64) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<EvaluationState> for u64 {
    fn from(e: EvaluationState) -> u64 {
        unsafe { std::mem::transmute(e) }
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches::assert_matches;

    use super::SpawnIter;
    use crate::search::search_state::Transition;

    #[test]
    fn test_complex() {
        let board = [[0, 1, 0, 1], [0, 2, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]];
        let board = crate::board::BoardAvx2::from_array(board).unwrap();

        let mut search_state = SpawnIter::from_board(board).unwrap();

        assert_eq!(search_state.current_board().to_array(), [
            [1, 1, 0, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);

        assert_matches!(search_state.next_spawn(), Transition::None);
        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 1, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);

        assert_matches!(search_state.next_spawn(), Transition::None);
        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 0, 1],
            [1, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);

        assert_matches!(search_state.next_spawn(), Transition::Switch);
        assert_eq!(search_state.current_board().to_array(), [
            [2, 1, 0, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);

        assert_matches!(search_state.next_spawn(), Transition::None);
        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 2, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);

        assert_matches!(search_state.next_spawn(), Transition::None);
        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 0, 1],
            [2, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);

        assert_matches!(search_state.next_spawn(), Transition::Done);
        assert_matches!(search_state.next_spawn(), Transition::Done);
        assert_matches!(search_state.next_spawn(), Transition::Done);
    }

    #[test]
    fn test_full() {
        let board = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
        let board = crate::board::BoardAvx2::from_array(board).unwrap();
        let search_state = SpawnIter::from_board(board);
        assert_matches!(search_state, None);
    }

    #[test]
    fn test_empty() {
        let board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let board = crate::board::BoardAvx2::from_array(board).unwrap();

        let mut search_state = SpawnIter::from_board(board).unwrap();

        let find = |search_state: &SpawnIter| {
            search_state
                .current_board()
                .to_array()
                .as_flattened()
                .iter()
                .take_while(|&a| *a == 0)
                .count()
        };

        for i in 0..15 {
            assert_eq!(find(&search_state), i);
            assert_matches!(search_state.next_spawn(), Transition::None);
        }

        assert_eq!(find(&search_state), 15);
        assert_matches!(search_state.next_spawn(), Transition::Switch);

        for i in 0..15 {
            assert_eq!(find(&search_state), i);
            assert_matches!(search_state.next_spawn(), Transition::None);
        }

        assert_eq!(find(&search_state), 15);
        assert_matches!(search_state.next_spawn(), Transition::Done);
        assert_matches!(search_state.next_spawn(), Transition::Done);
        assert_matches!(search_state.next_spawn(), Transition::Done);
    }
}
