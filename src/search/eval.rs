use std::fmt::Display;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Evaluation(pub i16);

impl Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Evaluation {
    pub const TERM: Self = Evaluation(-0x01FF);
    pub const WORST: Self = Evaluation(i16::MIN);
    pub const BEST: Self = Evaluation(0x01FF);

    pub fn as_fp(self) -> i32 {
        self.0 as i32
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct EvaluationState {
    pub numerator: i16,
    pub denominator: u16,
    pub best_move_eval: Evaluation,
    pub remaining_moves: u16,
}

impl EvaluationState {
    const NEW: EvaluationState = EvaluationState {
        numerator: 0,
        denominator: 0,
        best_move_eval: Evaluation::WORST,
        remaining_moves: 0,
    };

    pub fn new() -> EvaluationState {
        Self::NEW
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

    pub fn evaluate(self) -> Evaluation {
        debug_assert_eq!(self.remaining_moves, 0, "we have not tried all the moves");
        debug_assert!(self.denominator <= 3 * 15, "denominator not divisible by 3");
        Evaluation(self.numerator / self.denominator as i16)
    }

    pub fn switch(&mut self) {
        debug_assert!(self.denominator <= 15, "denominator not divisible by 3");
        self.numerator *= 2;
        self.denominator *= 2;
    }

    pub fn reset_moves(&mut self) {
        self.remaining_moves = 0;
        self.best_move_eval = Evaluation::WORST;
    }

    pub fn add_move(&mut self) {
        self.remaining_moves += 1;
    }
}

impl Default for EvaluationState {
    fn default() -> Self {
        Self::new()
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
