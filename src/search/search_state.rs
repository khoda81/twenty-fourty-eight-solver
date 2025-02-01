use std::arch::x86_64::*;

use crate::board::BoardAvx2;

#[derive(Debug)]
pub struct EvaluationState(__m128i);

#[derive(Debug)]
pub enum CurrentState {
    Done,
    One,
    Two,
}

impl EvaluationState {
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
            //let target = _mm_set1_epi8(a)

            crate::debug_println!("target: {target:?}");

            let target = _mm_loadu_si128(target.as_ptr().cast());
            let inner = _mm_blendv_epi8(inner, target, target);

            Some(Self(inner))
        }
    }

    pub fn next_spawn(&mut self) -> CurrentState {
        use std::mem::transmute as tb;

        let Self(mut inner) = *self;

        let shifted = unsafe { _mm_slli_epi16::<1>(inner) };
        let msb = unsafe { _mm_movemask_epi8(inner) as u16 };
        crate::debug_println!("msb:   {msb:016b}");
        let state = unsafe { _mm_movemask_epi8(shifted) as u16 };
        crate::debug_println!("state: {state:016b}");

        let filled_mask = msb | state.wrapping_sub(1);
        let filled_mask = if filled_mask == u16::MAX {
            // Create a mask to extract the shifted value at previous_spawn position
            inner = unsafe { _mm_blendv_epi8(inner, shifted, shifted) };
            msb
        } else {
            filled_mask
        };

        let next_spawn_idx = (filled_mask & !(1 << 15)).trailing_ones() as usize;
        let previous_spawn = (state | (1 << 15)).trailing_zeros() as usize;

        //let mut shuffle: [u8; 16] = std::array::from_fn(|i| i as u8);
        //shuffle[next_spawn_idx as usize & 15] = previous_spawn as u8;
        //shuffle[previous_spawn as usize & 15] = next_spawn_idx as u8;
        //
        //crate::debug_println!("shuffle: {shuffle:?}");
        //crate::debug_println!("inner:   {shuffle:?}");
        //
        //let shuffle = _mm_loadu_si128(shuffle.as_ptr().cast());
        //self.0 = _mm_shuffle_epi8(inner, shuffle);

        unsafe {
            let mut cells: [u8; 16] = tb(inner);
            cells.swap_unchecked(next_spawn_idx, previous_spawn);
            //cells[n] = cells[p];
            //cells[p] = 0;
            let state_mask = _mm_set1_epi8(0b00100000);
            let state_m = _mm_and_si128(self.0, state_mask);

            self.0 = _mm_loadu_si128(cells.as_ptr().cast());

            if state == 0 {
                CurrentState::Done
            } else if _mm_movemask_epi8(_mm_cmpeq_epi8(state_m, _mm_setzero_si128())) == 0xFFFF {
                CurrentState::Two
            } else {
                CurrentState::One
            }
        }
    }

    pub fn current_board(&self) -> BoardAvx2 {
        unsafe {
            let mask = _mm_set1_epi8(0b10011111_u8 as i8);
            BoardAvx2(_mm_and_si128(self.0, mask))
        }
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches::assert_matches;

    use super::EvaluationState;
    use crate::search::search_state::CurrentState;

    #[test]
    fn test_complex() {
        let board = [[0, 1, 0, 1], [0, 2, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]];
        let board = crate::board::BoardAvx2::from_array(board).unwrap();

        let mut search_state = EvaluationState::from_board(board).unwrap();

        assert_eq!(search_state.current_board().to_array(), [
            [1, 1, 0, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);
        assert_matches!(search_state.next_spawn(), CurrentState::One);

        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 1, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);
        assert_matches!(search_state.next_spawn(), CurrentState::One);

        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 0, 1],
            [1, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);
        assert_matches!(search_state.next_spawn(), CurrentState::One);

        assert_eq!(search_state.current_board().to_array(), [
            [2, 1, 0, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);
        assert_matches!(search_state.next_spawn(), CurrentState::Two);

        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 2, 1],
            [0, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);
        assert_matches!(search_state.next_spawn(), CurrentState::Two);

        assert_eq!(search_state.current_board().to_array(), [
            [0, 1, 0, 1],
            [2, 2, 2, 1],
            [2, 2, 2, 1],
            [1, 1, 1, 1]
        ]);
        assert_matches!(search_state.next_spawn(), CurrentState::Two);

        assert_matches!(search_state.next_spawn(), CurrentState::Done);
        assert_matches!(search_state.next_spawn(), CurrentState::Done);
        assert_matches!(search_state.next_spawn(), CurrentState::Done);
    }

    #[test]
    fn test_full() {
        let board = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
        let board = crate::board::BoardAvx2::from_array(board).unwrap();
        let search_state = EvaluationState::from_board(board);
        assert_matches!(search_state, None);
    }

    #[test]
    fn test_empty() {
        let board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let board = crate::board::BoardAvx2::from_array(board).unwrap();

        let mut search_state = EvaluationState::from_board(board).unwrap();

        let find = |search_state: &EvaluationState| {
            search_state
                .current_board()
                .to_array()
                .as_flattened()
                .iter()
                .take_while(|&a| *a == 0)
                .count()
        };

        for i in 0..16 {
            assert_eq!(find(&search_state), i);
            assert_matches!(search_state.next_spawn(), CurrentState::One);
        }

        for i in 0..16 {
            assert_eq!(find(&search_state), i);
            assert_matches!(search_state.next_spawn(), CurrentState::Two);
        }

        assert_matches!(search_state.next_spawn(), CurrentState::Done);
        assert_matches!(search_state.next_spawn(), CurrentState::Done);
        assert_matches!(search_state.next_spawn(), CurrentState::Done);
    }
}
