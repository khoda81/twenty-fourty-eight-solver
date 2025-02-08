use std::{
    fmt::Display,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use crate::board::BoardAvx2;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Evaluation(u16);

impl Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Evaluation {
    pub const TERMINAL: Self = Evaluation(0);
    pub const WORST: Self = Evaluation(0);
    pub const BEST: Self = Evaluation(0b0000_0101_1010_1111); // (2**16 - 1) / (3 * 15)

    pub fn new(eval: u16) -> Self {
        debug_assert!(
            Self(eval) <= Self::BEST,
            "eval should be less or equal to best eval"
        );

        Self(eval)
    }

    pub fn new_fraction(numerator: u16, denominator: u16) -> Self {
        if numerator >= denominator {
            Self::BEST
        } else {
            let eval = Self::BEST.as_u16() as u32 * numerator as u32 / denominator as u32;
            Self::new(eval as u16)
        }
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }

    pub fn as_u32(self) -> u32 {
        self.0 as u32
    }
}

impl Add<u16> for Evaluation {
    type Output = Self;

    fn add(self, rhs: u16) -> Self::Output {
        Self(self.0.saturating_add(rhs)).min(Self::BEST)
    }
}

impl AddAssign<u16> for Evaluation {
    fn add_assign(&mut self, rhs: u16) {
        *self = *self + rhs;
    }
}

impl Sub<u16> for Evaluation {
    type Output = Self;

    fn sub(self, rhs: u16) -> Self::Output {
        Self(self.0.saturating_sub(rhs)).max(Self::WORST)
    }
}

impl SubAssign<u16> for Evaluation {
    fn sub_assign(&mut self, rhs: u16) {
        *self = *self - rhs;
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct EvaluationState {
    // (best - lower_bound) * total_weight
    pub max_loss: u16,
    /// Sum of the brach evaluations so far
    pub numerator: u16,
    /// Sum of the branch weights so far
    pub denominator: u8,
    /// Number of moves remaining
    pub remaining_moves: u8,
    /// Evaluation of the best move
    pub best_move_eval: Evaluation,
}

impl EvaluationState {
    pub fn new(lower_bound: Evaluation, board: BoardAvx2) -> EvaluationState {
        EvaluationState {
            max_loss: (Evaluation::BEST.0 - lower_bound.0) * total_weight(board),
            numerator: 0,
            denominator: 0,
            best_move_eval: Evaluation::WORST,
            remaining_moves: 0,
        }
    }

    pub fn push_move_eval(&mut self, eval: Evaluation) -> bool {
        debug_assert!(self.remaining_moves > 0);
        self.best_move_eval = self.best_move_eval.max(eval);
        self.remaining_moves -= 1;

        let done = self.remaining_moves == 0;
        if done {
            self.push_spawn_eval(self.best_move_eval)
        }

        done
    }

    pub fn push_spawn_eval(&mut self, value: Evaluation) {
        self.numerator += value.0;
        self.denominator += 1;
    }

    pub fn evaluate(&self) -> Evaluation {
        debug_assert_eq!(self.remaining_moves, 0, "we have not tried all the moves");
        debug_assert!(0 < self.denominator, "denominator is 0!");
        debug_assert!(self.denominator <= 3 * 15, "is bigger than 45!");

        Evaluation::new(self.numerator / self.denominator as u16)
    }

    pub fn switch(&mut self) {
        debug_assert!(self.denominator <= 15, "denominator not divisible by 3");
        self.numerator *= 2;
        self.denominator *= 2;
    }

    pub fn next_branch(&mut self) {
        self.remaining_moves = 0;
        self.best_move_eval = Evaluation::WORST;
    }

    pub fn add_move(&mut self) {
        self.remaining_moves += 1;
    }

    pub fn required_lower_bound(&self) -> Evaluation {
        let best = Evaluation::BEST.as_u16();
        let loss = self.loss();

        let eval = (best + loss).saturating_sub(self.max_loss);
        debug_assert!(
            eval <= best,
            "loss={} is smaller than gain={loss}, {}/{}",
            self.max_loss,
            self.numerator,
            self.denominator
        );
        Evaluation::new(eval).max(self.best_move_eval)
    }

    pub fn upper_bound(&self, board: BoardAvx2) -> Evaluation {
        Evaluation::new(Evaluation::BEST.as_u16() - self.loss() / total_weight(board))
    }

    fn loss(&self) -> u16 {
        let best = Evaluation::BEST.as_u16();
        let weight = self.denominator as u16;

        weight * best - self.numerator
    }

    pub fn prunable(&self) -> bool {
        self.loss() >= self.max_loss
    }
}

fn total_weight(board: BoardAvx2) -> u16 {
    board.num_empty() as u16 * 3
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
