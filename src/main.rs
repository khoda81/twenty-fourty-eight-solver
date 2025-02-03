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

    #[arg(short, long)]
    depth: Option<i32>,

    #[arg(short, long, default_value = "random")]
    starting_pos: StartingPosition,

    #[arg(short, long)]
    /// Keep the evaluation cache instead of clearing every move
    persistent_cache: bool,

    #[arg(long)]
    /// Depth of search for the heuristic
    heuristic_depth: Option<u32>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    Play,
    Eval,
    SingleShot,
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
        .parse_default_env()
        .init();

    let board = match args.starting_pos {
        StartingPosition::Random => random_spawn(BoardAvx2::new().unwrap()).unwrap(),
        StartingPosition::Manual => interactive_board_editor(),
    };

    let mut mean_max = MeanMax::new();
    if let Some(depth) = args.depth {
        mean_max.search_constraint.set_depth(depth);
    }

    if let Some(heuristic_depth) = args.heuristic_depth {
        mean_max.heuristic_depth = heuristic_depth;
    }

    match args.mode {
        Mode::Play => play(&mut mean_max, board, args),
        Mode::Eval => evaluate_heuristic(&mut mean_max, board),
        Mode::SingleShot => {
            info!("Evaluating:\n{board}");

            let start = std::time::Instant::now();
            let (eval, best_move) = mean_max.best_move(board);
            let elapsed = start.elapsed();
            info!("Best move: {best_move}, Eval: {eval}");

            let iterations = mean_max.iteration_counter as f64;

            if mean_max.iteration_counter > 0 {
                let per_iteration = elapsed / mean_max.iteration_counter;
                info!(
                    "{iterations:.2e} iterations in {elapsed:.2?} ({per_iteration:?} per iteration)"
                );
            } else {
                info!("{iterations:.2e} iterations in {elapsed:.2?}");
            }

            let cache = mean_max.cache();
            info!(
                "Hit ratio: {:.3} ({}/{})",
                cache.hit_rate(),
                cache.hit_counter(),
                cache.lookup_counter()
            );

            let mut board = board;
            for move_idx in 0..4 {
                if best_move == move_idx {
                    board = board.swipe_right();
                }

                board = board.rotate_90();
            }

            if board.num_empty() == 0 {
                info!("Game over!\n{board}");
            }
        }
    }
}

fn play(mean_max: &mut MeanMax, mut board: BoardAvx2, args: Args) {
    loop {
        info!("Evaluating:\n{board}");

        let start = std::time::Instant::now();
        if !args.persistent_cache {
            mean_max.clear_cache();
        }

        let (eval, best_move) = mean_max.best_move(board);
        let elapsed = start.elapsed();
        info!("Best move: {best_move}, Eval: {eval}");

        let iterations = mean_max.iteration_counter as f64;

        if mean_max.iteration_counter > 0 {
            let per_iteration = elapsed / mean_max.iteration_counter;
            info!("{iterations:.2e} iterations in {elapsed:.2?} ({per_iteration:?} per iteration)");
        } else {
            info!("{iterations:.2e} iterations in {elapsed:.2?}");
        }

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
    let n_games = 1000; // TODO: make this a commandline arg

    log::info!("Evaluating heuristic!");
    let mut running_total = 0;

    let total: u32 = (1..n_games + 1)
        .map(|i| {
            let score = run_game(mean_max, board);
            running_total += score;

            log::info!(
                "Game {i}/{n_games}: {score} (avg: {:.2})",
                running_total as f64 / i as f64
            );

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
    let board_values = board::editor::grid_editor().unwrap();

    let board = BoardAvx2::from_array(board_values).unwrap();
    if board == BoardAvx2::new().unwrap() {
        random_spawn(board).unwrap()
    } else {
        board
    }
}
