use std::fmt::Display;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use number_prefix::NumberPrefix;
use twenty_fourty_eight_solver::{
    board::{self, BoardAvx2},
    search::{
        mean_max::{MeanMax, SearchConstraint},
        search_state::SpawnIter,
    },
};

#[derive(Parser, Debug)]
#[command(name = "2048 Solver", version, about = "A solver for the 2048 game", long_about = None)]
struct Args {
    #[arg(value_enum, default_value = "play")]
    mode: Mode,

    #[arg(short, long)]
    depth: Option<i32>,

    #[arg(short, long)]
    board_editor: bool,

    #[arg(short, long)]
    /// Keep the evaluation cache instead of clearing every move
    persistent_cache: bool,

    #[arg(long)]
    /// Depth of search for the heuristic
    heuristic_depth: Option<u32>,

    #[arg(long, default_value = "1000")]
    num_eval_games: u64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    Play,
    Eval,
    SingleShot,
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .parse_default_env()
        .init();

    let board = if args.board_editor {
        interactive_board_editor()
    } else {
        SpawnIter::new(BoardAvx2::new().unwrap())
            .unwrap()
            .random_spawn(&mut rand::rng())
    };

    let mut mean_max = MeanMax::new();

    if let Some(heuristic_depth) = args.heuristic_depth {
        mean_max.heuristic_depth = heuristic_depth;
    }

    match args.mode {
        Mode::Play => play(&mut mean_max, board, args),
        Mode::Eval => evaluate_heuristic(&mut mean_max, board, args),
        Mode::SingleShot => {
            info!("Evaluating:\n{board}");

            let best_move = search_best_move(&mut mean_max, SearchConstraint { board, depth: 7 });
            info!("Selected: {best_move}");
            let board = board.swipe_direction(best_move);
            info!("Board:\n{board}");

            if board.num_empty() == 0 {
                info!("Game over!\n{board}");
            }
        }
    }
}

fn play(mean_max: &mut MeanMax, board: BoardAvx2, args: Args) {
    let mut constraint = SearchConstraint { board, depth: 3 };

    if let Some(depth) = args.depth {
        constraint.set_depth(depth);
    }

    loop {
        if !args.persistent_cache {
            mean_max.clear_cache();
        }

        let best_move = search_best_move(mean_max, constraint);

        let spawn_iter = SpawnIter::new(constraint.board.swipe_direction(best_move));
        let Some(spawn_iter) = spawn_iter else { break };
        constraint.board = spawn_iter.random_spawn(&mut rand::rng());
    }

    info!("Game over!\n{board}");
}

struct Count(f64);
impl Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match NumberPrefix::decimal(self.0) {
            NumberPrefix::Standalone(val) => val.fmt(f),
            NumberPrefix::Prefixed(prefix, val) => {
                val.fmt(f)?;
                prefix.fmt(f)
            }
        }
    }
}

fn search_best_move(mean_max: &mut MeanMax, search_constraint: SearchConstraint) -> u16 {
    info!("Evaluating:\n{}", search_constraint.board);

    let start = std::time::Instant::now();

    let (eval, best_move) = mean_max.search(search_constraint);
    let elapsed = start.elapsed();
    info!("Best move: {best_move}, Eval: {eval}");

    let iterations = mean_max.iteration_counter as f64;

    if mean_max.iteration_counter > 0 {
        let per_iteration = elapsed / mean_max.iteration_counter;
        info!(
            "{:.0} iterations in {elapsed:.2?} ({per_iteration:?} per iteration)",
            Count(iterations)
        );
    } else {
        info!("{:.0} iterations in {elapsed:.2?}", Count(iterations));
    }

    let cache = mean_max.cache();
    info!(
        "Hit rate: {:.3} ({:.0}/{:.0})",
        cache.hit_rate(),
        Count(cache.hit_counter() as f64),
        Count(cache.lookup_counter() as f64),
    );

    best_move
}

fn evaluate_heuristic(mean_max: &mut MeanMax, board: BoardAvx2, args: Args) {
    log::info!("Evaluating heuristic!");
    let mut constraint = SearchConstraint { board, depth: 0 };

    if let Some(depth) = args.depth {
        constraint.set_depth(depth);
    }

    let style =
        ProgressStyle::with_template("{pos}/{len} Games [{wide_bar:^.cyan/blue}] Eta: {eta} {msg}")
            .unwrap()
            .progress_chars("=> ");

    let pb = ProgressBar::new(args.num_eval_games).with_style(style);
    let mut total = 0;

    for i in 1..args.num_eval_games + 1 {
        let score = run_game(mean_max, constraint);
        total += score;

        let message = format!("Avg: {:6.2}", total as f64 / i as f64);

        pb.set_message(message);
        pb.inc(1);
    }

    pb.finish();
    let eval = total as f64 / args.num_eval_games as f64;
    log::info!("Avg: {eval}");
}

fn run_game(mean_max: &mut MeanMax, mut constraint: SearchConstraint) -> u32 {
    let mut score = 0;

    loop {
        mean_max.clear_cache();

        let (_, best_move) = mean_max.search(constraint);

        let spawn_iter = SpawnIter::new(constraint.board.swipe_direction(best_move));
        let Some(spawn_iter) = spawn_iter else { break };
        constraint.board = spawn_iter.random_spawn(&mut rand::rng());
        score += 1;
    }

    score
}

fn interactive_board_editor() -> BoardAvx2 {
    let board_values = board::editor::grid_editor().unwrap();

    let board = BoardAvx2::from_array(board_values).unwrap();
    if board == BoardAvx2::new().unwrap() {
        SpawnIter::new(board)
            .unwrap()
            .random_spawn(&mut rand::rng())
    } else {
        board
    }
}
