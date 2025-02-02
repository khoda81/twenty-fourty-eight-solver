use rand::seq::IndexedRandom;
use twenty_fourty_eight_solver::{
    board::BoardAvx2,
    search::{
        mean_max::MeanMax,
        search_state::{SpawnIter, Transition},
    },
};

fn main() {
    //let cells = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 0]];
    let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    let mut board = BoardAvx2::from_array(cells).unwrap();
    let mut mean_max = MeanMax::new();

    loop {
        eprintln!("Evaluating:\n{board:?}");

        let start = std::time::Instant::now();
        let (best_move, eval) = mean_max.best_move(board, 9);
        let elapsed = start.elapsed();
        eprintln!("Best move: {best_move}, eval: {}", eval.0);

        eprintln!("Iterations: {}", mean_max.iteration_counter);
        eprintln!(
            "In {:?} ({:?} per iteration)",
            elapsed,
            elapsed / mean_max.iteration_counter
        );

        let cache = mean_max.cache();
        eprintln!(
            "Hit ratio: {:.3} ({}/{})",
            cache.hit_rate(),
            cache.hit_counter(),
            cache.lookup_counter()
        );

        for move_idx in 0..4 {
            if best_move == move_idx {
                board = board.swipe_right();
            }

            board = board.rotate_90();
        }

        let mut spawns = vec![];
        let Some(mut spawner) = SpawnIter::new(board) else {
            break;
        };
        let mut weight = 2;

        board = loop {
            for _ in 0..weight {
                spawns.push(spawner.current_board());
            }

            match spawner.next_spawn() {
                Transition::Done => break *spawns.choose(&mut rand::rng()).unwrap(),
                Transition::Switch => weight -= 1,
                Transition::None => {}
            }
        }
    }
}
