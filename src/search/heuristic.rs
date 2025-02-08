use crate::{
    board::BoardAvx2,
    search::{
        eval::Evaluation,
        node::{SpawnNode, Transition},
    },
};

use rand::Rng as _;

const N_GAMES: usize = 20;
const STEP_LIMIT: usize = 200;

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
        let mut game_board = board;
        let mut game_steps = 0;
        'game: for _ in 0..STEP_LIMIT {
            let Some(mut node) = SpawnNode::new(game_board) else {
                break;
            };

            let num_empty = game_board.num_empty();
            let steps = rng.random_range(0..num_empty * 3);
            if steps >= num_empty * 2 {
                // Spawn a two
                while let Transition::None = node.next_spawn() {}
            }

            for _ in 0..steps % 3 {
                node.next_spawn();
            }

            game_board = node.current_branch();
            for _ in 0..4 {
                if let Some(b) = game_board.checked_swipe_right() {
                    game_board = b;
                    game_steps += 1;
                    continue 'game;
                };
                game_board = game_board.rotate_90();
            }

            break;
        }

        eval_sum += game_steps * Evaluation::BEST.as_u16() as usize / STEP_LIMIT;
    }

    let eval = (eval_sum / N_GAMES).try_into().unwrap();

    Evaluation::new(eval)
}
