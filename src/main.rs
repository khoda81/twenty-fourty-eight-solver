use twenty_fourty_eight_solver::{board::BoardAvx2, search::mean_max::MeanMax};

fn main() {
    //let cells = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 0]];
    let cells = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    let board = BoardAvx2::from_array(cells).unwrap();
    let mut mean_max = MeanMax::new();
    let start = std::time::Instant::now();
    let (move_idx, eval) = mean_max.best_move(board, 3);
    let elapsed = start.elapsed();
    eprintln!(
        "Evaluated:\n{board:?}\nMove idx: {move_idx}, eval: {}",
        eval.0
    );

    eprintln!("Iterations: {}", mean_max.counter);
    eprintln!(
        "In {:?} ({:?} per iteration)",
        elapsed,
        elapsed / mean_max.counter
    );
}
