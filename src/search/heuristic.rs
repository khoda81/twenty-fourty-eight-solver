use crate::{
    board::BoardAvx2,
    search::{eval::Evaluation, node::SpawnNode},
};

const N_GAMES: usize = 20;
const STEP_LIMIT: u16 = 200;

pub fn heuristic(board: BoardAvx2) -> Evaluation {
    let rng = &mut rand::rng();
    //let mut eval = Evaluation::BEST.as_u16();

    //for i in 0..2 {
    //    let Some(mut node) = SpawnNode::new(board) else {
    //        eval >>= 3 - i as i16;
    //        break;
    //    };
    //
    //    let num_empty = board.num_empty();
    //    let steps = rng.random_range(0..num_empty * 3);
    //    if steps >= num_empty * 2 {
    //        // Spawn a two
    //        while let Transition::None = node.next_spawn() {}
    //    }
    //
    //    for _ in 0..steps % 3 {
    //        node.next_spawn();
    //    }
    //
    //    match node.current_branch().rotate_90().checked_swipe_right() {
    //        Some(b) => board = b,
    //        None => break,
    //    };
    //}
    //
    //
    //let num_empty = board.num_empty();
    //
    //const OTHER: u32 = 10;
    //eval = eval.saturating_sub((1 << OTHER) - (1 << num_empty.min(OTHER)));

    let mut eval_sum = 0;
    for _ in 0..N_GAMES {
        let game_steps = random_rollout(rng, board);
        eval_sum += game_steps.as_u16() as usize;
    }

    let eval = (eval_sum / N_GAMES).try_into().unwrap();

    Evaluation::new(eval)
}

pub fn random_rollout(rng: &mut impl rand::Rng, mut board: BoardAvx2) -> Evaluation {
    'game: for game_steps in 0..STEP_LIMIT {
        // Try to create an spawn node (fails if the board is full)
        match SpawnNode::random_spawned(rng, board) {
            None => return Evaluation::new(game_steps),
            Some(spawned) => board = spawned,
        };

        // Try to do a move
        for rot in board.rotate_90().rotations() {
            if let Some(swiped) = rot.checked_swipe_right() {
                board = swiped;
                continue 'game;
            }
        }

        return Evaluation::new(game_steps);
    }

    Evaluation::BEST
}
