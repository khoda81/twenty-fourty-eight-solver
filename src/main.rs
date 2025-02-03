use clap::{Parser, ValueEnum};
use log::info;
use rand::seq::IndexedRandom;
use twenty_fourty_eight_solver::{
    board::{self, BoardAvx2},
    search::{
        mean_max::MeanMax,
        search_state::{SpawnIter, Transition},
    },
};

#[derive(Parser, Debug)]
#[command(name = "2048 Solver", version, about = "A solver for the 2048 game", long_about = None)]
struct Args {
    #[arg(value_enum, default_value = "play")]
    mode: Mode,

    #[arg(short, long, default_value_t = 3)]
    depth: i32,

    #[arg(short, long, default_value = "random")]
    starting_pos: StartingPosition,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    Play,
    Eval,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum StartingPosition {
    Random,
    Manual,
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    let board = match args.starting_pos {
        StartingPosition::Random => random_spawn(BoardAvx2::new().unwrap()).unwrap(),
        StartingPosition::Manual => interactive_board_editor(),
    };

    let mut mean_max = MeanMax::new();
    mean_max.search_constraint.set_depth(args.depth);

    match args.mode {
        Mode::Play => play(&mut mean_max, board),
        Mode::Eval => evaluate_heuristic(&mut mean_max, board),
    }
}

pub fn play(mean_max: &mut MeanMax, mut board: BoardAvx2) {
    loop {
        info!("Evaluating:\n{board}");

        let start = std::time::Instant::now();
        mean_max.clear_cache();

        let (eval, best_move) = mean_max.best_move(board);
        let elapsed = start.elapsed();
        info!("Best move: {best_move}, Eval: {eval}");

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

pub fn evaluate_heuristic(mean_max: &mut MeanMax, board: BoardAvx2) {
    let n_games = 1000;

    let total: u32 = (0..n_games)
        .map(|i| {
            let score = run_game(mean_max, board);
            log::debug!("Game {i}/{n_games}: {score}");
            score
        })
        .sum();

    let eval = total as f64 / n_games as f64;
    log::info!("Avg: {eval} ({total}/{n_games})");
}

fn run_game(mean_max: &mut MeanMax, mut board: BoardAvx2) -> u32 {
    let mut score = 0;

    loop {
        mean_max.clear_cache();

        let (_, best_move) = mean_max.best_move(board);

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

fn interactive_board_editor() -> BoardAvx2 {
    println!("Enter the board as a 4x4 grid (use space-separated values, 0 for empty):");
    let board_values = board::editor::grid_editor().unwrap();

    let board = BoardAvx2::from_array(board_values).unwrap();
    if board == BoardAvx2::new().unwrap() {
        random_spawn(board).unwrap()
    } else {
        board
    }
}
