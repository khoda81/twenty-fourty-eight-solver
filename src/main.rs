use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use number_prefix::NumberPrefix;
use twenty_fourty_eight_solver::{
    board::{BoardAvx2, editor},
    search::{
        eval::Evaluation,
        mcts::{MonteCarloTreeSearch as MCTS, SearchConstraint as MctsConstraint},
        mean_max::{MeanMax, SearchConstraint},
        node::SpawnNode,
    },
};

fn parse_duration(arg: &str) -> Result<std::time::Duration, std::num::ParseFloatError> {
    let seconds = arg.parse()?;
    Ok(std::time::Duration::from_secs_f64(seconds))
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Algorithm {
    MeanMax,
    Mcts,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    Eval,
    #[default]
    SingleGame,
    SingleShot,
}

#[derive(Parser, Debug)]
#[command(name = "2048 Solver", version, about = "A solver for the 2048 game", long_about = None)]
struct Args {
    #[arg(value_enum, short, long, default_value = "single-game")]
    mode: Mode,

    #[arg(value_enum, short, long, default_value = "mcts")]
    algorithm: Algorithm,

    #[arg(short, long)]
    depth: Option<i32>,

    #[arg(short, long)]
    board_editor: bool,

    #[arg(short, long)]
    /// Keep the evaluation cache instead of clearing every move
    persistent_cache: bool,

    #[arg(short, long, default_value = "1000")]
    num_eval_games: u64,

    #[arg(short='t', long, value_parser=parse_duration)]
    search_time: Option<Duration>,

    #[arg(short = 'C')]
    exploration_constant: Option<f64>,
}

trait Search {
    type Constraint;

    fn search_best_move(&mut self, constraint: Self::Constraint) -> (Evaluation, u16);
    fn constraint_from_args(board: BoardAvx2, args: &Args) -> Self::Constraint;
    fn clear(&mut self);
}

impl Search for MeanMax {
    type Constraint = SearchConstraint;

    fn search_best_move(&mut self, constraint: Self::Constraint) -> (Evaluation, u16) {
        let start = Instant::now();
        let (eval, best_move) = self.search_flexible(constraint);
        let elapsed = start.elapsed();

        let iterations = self.iteration_counter as f64;

        if self.iteration_counter > 0 {
            let per_iteration = elapsed / self.iteration_counter;
            log::debug!(
                "{:.0} iterations in {elapsed:.2?} ({per_iteration:.2?} per iteration)",
                Count(iterations)
            );
        } else {
            log::debug!("{:.0} iterations in {elapsed:.2?}", Count(iterations));
        }

        let cache = self.eval_cache();
        log::debug!(
            "Eval cache hit rate: {:.3} ({:.0}/{:.0})",
            cache.hit_rate(),
            Count(cache.hit_counter() as f64),
            Count(cache.lookup_counter() as f64),
        );

        let cache = self.prune_cache();
        log::debug!(
            "Prune cache hit rate: {:.3} ({:.0}/{:.0})",
            cache.hit_rate(),
            Count(cache.hit_counter() as f64),
            Count(cache.lookup_counter() as f64),
        );

        (eval, best_move)
    }

    fn constraint_from_args(board: BoardAvx2, args: &Args) -> Self::Constraint {
        SearchConstraint {
            board,
            depth: args.depth,
            deadline: args.search_time.map(|d| Instant::now() + d),
        }
    }

    fn clear(&mut self) {
        self.clear_cache();
    }
}

impl Search for MCTS {
    type Constraint = MctsConstraint;

    fn search_best_move(&mut self, mut constraint: Self::Constraint) -> (Evaluation, u16) {
        let start = Instant::now();
        let search_duration = constraint.deadline.duration_since(start);
        log::debug!("Searching for {search_duration:.2?}");
        let deadline = constraint.deadline;
        let mut iterations = 0;

        loop {
            let next_print = Instant::now() + Duration::from_secs_f32(0.1);
            constraint.deadline = deadline.min(next_print);

            let ((eval, best_move), iters) = self.search(constraint);
            let elapsed = start.elapsed();

            iterations += iters;
            let lookups = self.cache_lookup_counter as f64;
            let hits = self.cache_hit_counter as f64;
            let per_second = iterations as f64 / elapsed.as_secs_f64();

            log::debug!(
                "Best move: {best_move}, Eval: {eval}, {:.0} iters in {elapsed:.2?} ({:.0}/s), Hit rate: {:.3} ({:.0}/{:.0})",
                Count(iterations as f64),
                Count(per_second),
                hits / lookups,
                Count(hits),
                Count(lookups),
            );

            if log::log_enabled!(log::Level::Debug) {
                let moves = self.children(constraint.board).unwrap();
                log::debug!("Moves: {moves:?}");
            }

            if Instant::now() > deadline {
                break (eval, best_move);
            }
        }
    }

    fn constraint_from_args(board: BoardAvx2, args: &Args) -> Self::Constraint {
        let deadline = args
            .search_time
            .map(|d| Instant::now() + d)
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(1));

        MctsConstraint { board, deadline }
    }

    fn clear(&mut self) {
        self.clear_cache();
    }
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .parse_default_env()
        .init();

    let board = args.board_editor.then(interactive_board_editor);

    match args.algorithm {
        Algorithm::MeanMax => {
            log::debug!("Using MeanMax for search");
            dispatch_mode(&mut MeanMax::new(), board, args)
        }

        Algorithm::Mcts => {
            log::debug!("Using MonteCarloTreeSearch for search");
            if args.depth.is_some() {
                log::warn!("Depth is ignored as its not supported with monte-carlo");
            }

            let mut mcts = MCTS::new();
            if let Some(exploration_constant) = args.exploration_constant {
                mcts.exploration_rate = exploration_constant;
            }

            dispatch_mode(&mut mcts, board, args)
        }
    }
}

fn random_board() -> BoardAvx2 {
    SpawnNode::random_spawned(&mut rand::rng(), BoardAvx2::new().unwrap()).unwrap()
}

fn dispatch_mode<S: Search>(searcher: &mut S, board: Option<BoardAvx2>, args: Args) {
    match args.mode {
        Mode::SingleGame => play(searcher, board.unwrap_or_else(random_board), args),
        Mode::Eval => evaluate(searcher, board, args),
        Mode::SingleShot => {
            let board = board.unwrap_or_else(random_board);
            let constraint = S::constraint_from_args(board, &args);
            let (_eval, best_move) = searcher.search_best_move(constraint);
            log::info!("Selected: {best_move}");

            let board = board.swipe_direction(best_move);
            log::info!("Board:\n{board}");

            if board.num_empty() == 0 {
                log::info!("Game over!\n{board}");
            }
        }
    }
}

fn play<S: Search>(searcher: &mut S, mut board: BoardAvx2, args: Args) {
    loop {
        log::info!("Board:\n{board}");
        if !args.persistent_cache {
            log::debug!("Clearing cache!");
            searcher.clear();
        }

        let constraint = S::constraint_from_args(board, &args);
        let (eval, best_move) = searcher.search_best_move(constraint);

        log::info!("Best move: {best_move}, Eval: {eval}");

        let spawn_iter = SpawnNode::new(board.swipe_direction(best_move));
        let Some(spawn_iter) = spawn_iter else { break };
        board = spawn_iter.random_spawn(&mut rand::rng());
    }

    log::info!("Game over!\n{board}");
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

fn evaluate<S: Search>(searcher: &mut S, board: Option<BoardAvx2>, args: Args) {
    if let Some(board) = board {
        log::info!("Evaluating on:\n{board}");
    }

    let style =
        ProgressStyle::with_template("{pos}/{len} Games [{wide_bar:^.cyan/8}] Eta: {eta} {msg}")
            .unwrap()
            .progress_chars("=>.");

    let pb = ProgressBar::new(args.num_eval_games).with_style(style);
    let mut total = 0;

    for i in 1..args.num_eval_games + 1 {
        total += run_game(searcher, board.unwrap_or_else(random_board), &args);

        let message = format!("Avg: {:6.2}", total as f64 / i as f64);

        pb.set_message(message);
        pb.inc(1);
    }

    pb.finish();
    let eval = total as f64 / args.num_eval_games as f64;
    log::info!("Avg: {eval}");

    // Print ONLY the final score to stdout
    println!("{eval}");
}

fn run_game<S: Search>(searcher: &mut S, mut board: BoardAvx2, args: &Args) -> u32 {
    let mut score = 0;
    searcher.clear();

    loop {
        searcher.clear();

        let constraint = S::constraint_from_args(board, args);
        let (_, best_move) = searcher.search_best_move(constraint);

        let spawn_iter = SpawnNode::new(board.swipe_direction(best_move));
        let Some(spawn_iter) = spawn_iter else { break };
        board = spawn_iter.random_spawn(&mut rand::rng());
        score += 1;
    }

    score
}

fn interactive_board_editor() -> BoardAvx2 {
    let board_values = editor::grid_editor().unwrap();
    let board = BoardAvx2::from_array(board_values).unwrap();

    if board == BoardAvx2::new().unwrap() {
        SpawnNode::new(board)
            .unwrap()
            .random_spawn(&mut rand::rng())
    } else {
        board
    }
}
