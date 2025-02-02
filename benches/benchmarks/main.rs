use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use itertools::Itertools as _;
use twenty_fourty_eight_solver::{
    board::{BoardAvx2, test_utils},
    search::search_state::{SpawnIter, Transition},
};

/// Generate a vector of random boards for benchmarking.
fn generate_boards(count: usize) -> Vec<[[u8; 4]; 4]> {
    (0..16)
        .flat_map(|filled| {
            (0..filled).cartesian_product(0..count).map(move |(dup, _)|
            // Generate a random board with the specified number of filled cells
            test_utils::generate_random_board::<4, 4>(filled, dup))
        })
        .collect()
}

/// Benchmark the SIMD-optimized implementation.
fn bench_swipe(c: &mut Criterion) {
    const COUNT: usize = 100;

    let mut group = c.benchmark_group("swipe");

    let boards = generate_boards(COUNT);
    let simd_boards = boards
        .iter()
        .cloned()
        .map(BoardAvx2::from_array)
        .map(Result::unwrap)
        .collect_vec();

    group.throughput(Throughput::Elements(boards.len() as u64));

    group.bench_function("baseline_swipe", |b| {
        b.iter(|| {
            for &board in &boards {
                black_box(test_utils::baseline_swipe(board));
            }
        });
    });

    group.bench_function("simd_swipe", |b| {
        b.iter(|| {
            for board in &simd_boards {
                black_box(board.swipe_right());
            }
        });
    });
}

fn bench_rotate(c: &mut Criterion) {
    const COUNT: usize = 100;

    let mut group = c.benchmark_group("rotate_90");

    let boards = generate_boards(COUNT);
    let simd_boards = boards
        .iter()
        .cloned()
        .map(BoardAvx2::from_array)
        .map(Result::unwrap)
        .collect_vec();

    group.throughput(Throughput::Elements(boards.len() as u64));

    group.bench_function("simd_rotate_90", |b| {
        b.iter(|| {
            for board in simd_boards.iter() {
                black_box(board.rotate_90());
            }
        });
    });
}
/// Benchmark iterating over all boards and spawning until reaching the "Done" state.
fn bench_spawn_until_done(c: &mut Criterion) {
    const COUNT: usize = 100;

    let mut group = c.benchmark_group("spawn_until_done");

    let boards = generate_boards(COUNT);
    let simd_boards = boards
        .iter()
        .cloned()
        .map(BoardAvx2::from_array)
        .map(Result::unwrap)
        .collect_vec();

    group.throughput(Throughput::Elements(boards.len() as u64));

    group.bench_function("spawn_until_done", |b| {
        b.iter(|| {
            for board in &simd_boards {
                let Some(mut state) = SpawnIter::new(*board) else {
                    continue;
                };

                while !matches!(state.next_spawn(), Transition::Done) {
                    black_box(state.current_board());
                }
            }
        });
    });
}

/// Benchmark merging and spawning interleaved N times.
fn bench_swipe_and_spawn_interleaved(c: &mut Criterion) {
    const COUNT: usize = 100;
    const N: usize = 10;

    let mut group = c.benchmark_group("merge_and_spawn_interleaved");

    let boards = generate_boards(COUNT);
    let simd_boards = boards
        .iter()
        .cloned()
        .map(BoardAvx2::from_array)
        .map(Result::unwrap)
        .collect_vec();

    group.throughput(Throughput::Elements(boards.len() as u64));

    group.bench_function("merge_and_spawn_interleaved", |b| {
        b.iter(|| {
            for board in &simd_boards {
                let mut board = *board;

                for _ in 0..N {
                    board = board.swipe_right().rotate_90();
                    let Some(state) = SpawnIter::new(board) else {
                        continue;
                    };

                    board = state.current_board();
                }

                black_box(board);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_swipe,
    bench_rotate,
    bench_spawn_until_done,
    bench_swipe_and_spawn_interleaved
);
criterion_main!(benches);
