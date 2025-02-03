use log::info;
use rand::seq::IndexedRandom;
use twenty_fourty_eight_solver::{
    board::BoardAvx2,
    search::{
        mean_max::MeanMax,
        search_state::{SpawnIter, Transition},
    },
};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    play();
    //evaluate_heuristic();
}

fn play() {
    //let cells = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 0]];
    //let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    //let cells = [[1, 4, 7, 3], [1, 2, 1, 6], [0, 1, 0, 3], [0, 0, 0, 0]];
    //let mut board = BoardAvx2::from_array(cells).unwrap();

    let mut board = random_spawn(BoardAvx2::new().unwrap()).unwrap();
    let mut mean_max = MeanMax::new();

    loop {
        info!("Evaluating:\n{board}");

        let start = std::time::Instant::now();
        mean_max.search_constraint.set_depth(3);
        mean_max.clear_cache();

        let (best_move, eval) = mean_max.best_move(board);
        eprintln!("\x1B[2J\x1B[1;1H");
        let elapsed = start.elapsed();
        info!("Best move: {best_move}, Eval: {}", eval.0);

        info!("Iterations: {}", mean_max.iteration_counter);
        info!(
            "In {:?} ({:?} per iteration)",
            elapsed,
            elapsed / mean_max.iteration_counter
        );

        let cache = mean_max.cache();
        info!(
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

        let Some(new_board) = random_spawn(board) else {
            break;
        };
        board = new_board;
    }

    info!("Game over!\n{board}");
}

fn random_spawn(board: BoardAvx2) -> Option<BoardAvx2> {
    let mut spawns = vec![];
    let mut spawner = SpawnIter::new(board)?;
    let mut weight = 2;

    loop {
        for _ in 0..weight {
            spawns.push(spawner.current_board());
        }

        match spawner.next_spawn() {
            Transition::Done => break spawns.choose(&mut rand::rng()).copied(),
            Transition::Switch => weight -= 1,
            Transition::None => {}
        }
    }
}

fn evaluate_heuristic() {
    let n_games = 1000;
    let mut mean_max = MeanMax::new();
    mean_max.search_constraint.set_depth(0);

    let total: u32 = (0..n_games)
        .map(|i| {
            let score = run_game(&mut mean_max);
            log::debug!("Game {i}/{n_games}: {score}");
            score
        })
        .sum();

    let eval = total as f64 / n_games as f64;
    log::info!("Avg: {eval} ({total}/{n_games})");
}

fn run_game(mean_max: &mut MeanMax) -> u32 {
    let mut board = random_spawn(BoardAvx2::new().unwrap()).unwrap();
    let mut score = 0;

    loop {
        mean_max.clear_cache();

        let (best_move, _) = mean_max.best_move(board);

        for move_idx in 0..4 {
            if best_move == move_idx {
                board = board.swipe_right();
            }

            board = board.rotate_90();
        }

        let Some(new_board) = random_spawn(board) else {
            break;
        };

        score += 1;
        board = new_board;
    }

    score
}
