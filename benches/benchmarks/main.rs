use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use itertools::Itertools as _;
use twenty_fourty_eight_solver::board::{BoardAvx2, test_utils};

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

criterion_group!(benches, bench_swipe);
criterion_main!(benches);
