use std::mem::transmute as tb;
use std::{
    arch::x86_64::*,
    fmt::{self, Write},
};

use thiserror::Error;

#[derive(Clone, Copy)]
pub struct BoardAvx2(__m128i);

/// Load the lookup table as an AVX2 register.
unsafe fn load_fill_table() -> __m256 {
    const E: i8 = -0x80; // Empty
    const PERM_LOOKUP: [[i8; 4]; 8] = [
        [0, E, E, E], // x000
        [3, 0, E, E], // x001
        [2, 0, E, E], // x010
        [3, 2, 0, E], // x011
        [1, 0, E, E], // x100
        [3, 1, 0, E], // x101
        [2, 1, 0, E], // x110
        [3, 2, 1, 0], // x111
    ];

    let lookup_ptr = PERM_LOOKUP.as_flattened().as_ptr();
    unsafe { _mm256_loadu_ps(lookup_ptr as *const f32) }
}

/// Load the lookup table as an AVX2 register.
unsafe fn load_merge_table() -> __m256 {
    const E: i8 = 0x80_u8 as i8; // Empty
    const O: i8 = 0x05_u8 as i8; // Cell[1] + 1
    const T: i8 = 0x06_u8 as i8; // Cell[2] + 1
    const PERM_LOOKUP: [[i8; 4]; 8] = [
        [0, 1, 2, 3], // x000
        [O, 2, 3, E], // x001
        [0, O, 3, E], // x010
        [O, 2, 3, E], // x011
        [0, 1, T, E], // x100
        [O, T, E, E], // x101
        [0, O, 3, E], // x110
        [O, T, E, E], // x111
    ];

    let lookup_ptr = PERM_LOOKUP.as_flattened().as_ptr();
    unsafe { _mm256_loadu_ps(lookup_ptr as *const f32) }
}

/// Interleave the board data for mask extraction.
unsafe fn interleave_board(board_mm256: __m256i) -> __m256i {
    let indices = [
        0xFFFFFFFF_00010203_u64,
        0xFFFFFFFF_04050607_u64,
        0xFFFFFFFF_08090A0B_u64,
        0xFFFFFFFF_0C0D0E0F_u64,
    ];

    unsafe {
        // Load into a YMM register
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(board_mm256, indices)
    }
}

/// Broadcast the mask into an AVX2 register.
unsafe fn mask_to_idx(mask: i32) -> __m256i {
    let indices = [
        0xFFFFFFFF_FFFFFF00_u64,
        0xFFFFFFFF_FFFFFF01_u64,
        0xFFFFFFFF_FFFFFF02_u64,
        0xFFFFFFFF_FFFFFF03_u64,
    ];

    unsafe {
        let mask_broadcast = _mm256_set1_epi32(mask);

        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(mask_broadcast, indices)
    }
}

unsafe fn merge_mask_to_idx(mask: i32) -> __m256i {
    let indices = [
        0xFFFFFF00_FFFFFF00_u64,
        0xFFFFFF01_FFFFFF01_u64,
        0xFFFFFF02_FFFFFF02_u64,
        0xFFFFFF03_FFFFFF03_u64,
    ];

    unsafe {
        let mask_broadcast = _mm256_set1_epi32(mask);

        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(mask_broadcast, indices)
    }
}

/// Broadcast the mask into an AVX2 register.
unsafe fn extract_board(compacted_board: __m256i) -> __m128i {
    fn f<T: std::fmt::LowerHex>(name: &str, inp: &[T]) {
        eprint!("{name:15}: [");

        eprint!("{:#010x}", inp[0]);
        for idx in &inp[1..] {
            eprint!(", {idx:#010x}");
        }
        eprintln!("]");
    }

    unsafe {
        let lo = _mm256_castsi256_si128(compacted_board); // Lower 128 bits (indices [0-3])
        let hi = _mm256_extracti128_si256::<1>(compacted_board); // Upper 128 bits (indices [4-7])

        // Create a shuffle mask for gathering [0, 2, 3] from lo and [5] from hi
        let lo = _mm_shuffle_epi32::<0b11001100>(lo); // [0, 2, 3, X]
        let hi = _mm_shuffle_epi32::<0b11001100>(hi); // Extract index 5

        //f("lo", &tb::<_, [u32; 4]>(lo));
        //f("hi", &tb::<_, [u32; 4]>(hi));
        //let compacted_board = _mm256_castsi256_ps(compacted_board);
        //let out = _mm256_permute_ps::<0b00011011>(compacted_board);
        //let out = _mm256_castps_si256(out);
        //_mm256_castsi256_si128(out)
        _mm_blend_epi32::<0b1100>(lo, hi)
    }
}

/// Broadcast the mask into an AVX2 register.
unsafe fn get_merge_target(compacted_board: __m256i) -> __m256i {
    let indices = [
        0x03020100_03020100_u64,
        0x0B0A0908_0B0A0908_u64,
        0x03020100_03020100_u64,
        0x0B0A0908_0B0A0908_u64,
    ];

    unsafe {
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        let ones = _mm256_set1_pd(f64::from_bits(0x0101010100000000));
        let ones = _mm256_castpd_si256(ones);
        let zeros = _mm256_setzero_si256();

        let result = _mm256_shuffle_epi8(compacted_board, indices);
        let ones = _mm256_blendv_epi8(zeros, ones, result);
        _mm256_add_epi8(result, ones)
    }
}

/// Get row permutations based on the broadcasted mask.
unsafe fn lookup(lookup_mm256: __m256, mask_broadcast: __m256i) -> __m256 {
    unsafe { _mm256_permutevar8x32_ps(lookup_mm256, mask_broadcast) }
}

/// Add row offsets to the row permutations.
unsafe fn add_offsets(row_permutations: __m256) -> __m256i {
    unsafe {
        let row_offsets = _mm256_set_epi64x(
            0x0C0C0C0C0C0C0C0C,
            0x0808080808080808,
            0x0404040404040404,
            0x0000000000000000,
        );

        let row_permutations = _mm256_castps_si256(row_permutations);
        _mm256_add_epi32(row_permutations, row_offsets)
    }
}

/// Add row offsets for merge
unsafe fn add_merge_offsets(row_permutations: __m256) -> __m256i {
    unsafe {
        let row_offsets = _mm256_set_epi64x(
            0x0808080808080808,
            0x0000000000000000,
            0x0808080808080808,
            0x0000000000000000,
        );

        let row_permutations = _mm256_castps_si256(row_permutations);
        _mm256_add_epi32(row_permutations, row_offsets)
    }
}

/// Shift cells to right for comparison mask
unsafe fn comparison_target(board_mm256: __m256i) -> __m256i {
    //let indices = [
    //    0xFFFFFFFF_020100FF_u64,
    //    0xFFFFFFFF_050403FF_u64,
    //    0xFFFFFFFF_090A0BFF_u64,
    //    0xFFFFFFFF_0D0E0FFF_u64,
    //];
    let indices = [
        0xFFFFFFFF_FF030201_u64,
        0xFFFFFFFF_FF0B0A09_u64,
        0xFFFFFFFF_FF030201_u64,
        0xFFFFFFFF_FF0B0A09_u64,
    ];

    unsafe {
        // Load into a YMM register
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(board_mm256, indices)
    }
}

/// Generate comparison mask
unsafe fn compare_with_next(compacted_board_mm256: __m256i) -> i32 {
    fn f<T: std::fmt::LowerHex>(name: &str, inp: &[T]) {
        eprint!("{name:15}: [");

        eprint!("{:#010x}", inp[0]);
        for idx in &inp[1..] {
            eprint!(", {idx:#010x}");
        }
        eprintln!("]");
    }

    unsafe {
        //f("compacted", &tb::<_, [u32; 8]>(compacted_board_mm256));
        let target = comparison_target(compacted_board_mm256);
        //f("compare_target", &tb::<_, [u32; 8]>(target));

        let mask_256 = _mm256_cmpeq_epi8(compacted_board_mm256, target);

        _mm256_movemask_epi8(mask_256)
    }
}

/// # Safety
/// This function uses unsafe SIMD intrinsics. The caller must ensure that the
/// target CPU supports AVX2 and SSSE3 instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "ssse3")]
unsafe fn remove_msb(cells: __m128i) -> __m128i {
    unsafe {
        let zeros = _mm_setzero_si128();
        let m = _mm_set1_epi8(0x7f);
        let b = _mm_and_si128(cells, m);
        _mm_blendv_epi8(zeros, b, cells)
    }
}

/// # Safety
/// This function uses unsafe SIMD intrinsics. The caller must ensure that the
/// target CPU supports AVX2 and SSSE3 instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "ssse3")]
unsafe fn set_msb(board: __m128i) -> __m128i {
    unsafe {
        let zero = _mm_setzero_si128();

        // Compare for non-zero elements
        let zero_mask = _mm_cmpeq_epi8(board, zero); // 0xFF for zeros

        // Set MSB based on mask
        let msb = _mm_set1_epi8(0x80_u8 as i8);
        // MSB set in every byte
        _mm_or_si128(_mm_andnot_si128(zero_mask, msb), board)
    }
}

/// # Safety
/// This function uses unsafe SIMD intrinsics. The caller must ensure that the
/// target CPU supports AVX2 and SSSE3 instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "ssse3")]
unsafe fn swipe_right_simd(board_mm128: __m128i) -> __m128i {
    fn f<T: std::fmt::LowerHex>(name: &str, inp: &[T]) {
        eprint!("{name:15}: [");

        eprint!("{:#010x}", inp[0]);
        for idx in &inp[1..] {
            eprint!(", {idx:#010x}");
        }
        eprintln!("]");
    }

    fn fb<T: std::fmt::Binary>(name: &str, inp: &[T]) {
        eprint!("{name:15}: [");

        eprint!("{:#010b}", inp[0]);
        for idx in &inp[1..] {
            eprint!(", {idx:#010b}");
        }
        eprintln!("]");
    }

    unsafe {
        use std::mem::transmute as tb;

        let board_mm256 = _mm256_set_m128i(board_mm128, board_mm128); // u8x4x4x2
        let interleaved = interleave_board(board_mm256); // x x x x 0 1 2 3 x x x ...

        //f("interleaved", &tb::<_, [u32; 8]>(interleaved));
        //eprintln!("interleaved: {:?}", tb::<_, [u32; 8]>(interleaved));
        let fill_mask = _mm256_movemask_epi8(interleaved);
        let fill_pattern_idx = mask_to_idx(fill_mask); // u32x4

        //eprintln!("fill_mask: {:032b}", fill_mask);
        //fb("fill_pattern_id", &tb::<_, [u32; 8]>(fill_pattern_idx));
        //eprintln!(
        //    "fill_pattern_idx: {:?}",
        //    tb::<_, [u32; 8]>(interleaved).map(|i| i & 7)
        //);

        let lookup_mm256 = load_fill_table();
        let row_permutations = lookup(lookup_mm256, fill_pattern_idx);
        let permutations = add_offsets(row_permutations);
        let compacted_board = _mm256_shuffle_epi8(board_mm256, permutations);

        let eq_mask = compare_with_next(compacted_board);
        let merge_pattern_idx = merge_mask_to_idx(eq_mask);
        //eprintln!("eq_mask: {eq_mask:032b}");
        //fb("eq_mask", &tb::<_, [u32; 8]>(merge_pattern_idx));
        let merge_lookup = load_merge_table();
        let merge_pattern = lookup(merge_lookup, merge_pattern_idx);
        //f("merge_pattern", &tb::<_, [u32; 8]>(merge_pattern));
        let merge_target_idx = add_merge_offsets(merge_pattern);
        //f("merge_target_id", &tb::<_, [u32; 8]>(merge_target_idx));

        let merge_target = get_merge_target(compacted_board);
        //f("merge_target", &tb::<_, [u32; 8]>(merge_target));
        let merged = _mm256_shuffle_epi8(merge_target, merge_target_idx);
        //f("merged", &tb::<_, [u32; 8]>(merged));

        //eprintln!("merge_pattern: {:?}", tb::<_, [u32; 8]>(merge_pattern_idx));
        //eprintln!(
        //    "merge_pattern: {:?}",
        //    tb::<_, [u32; 8]>(merge_pattern_idx).map(|i| i & 7)
        //);
        //
        //eprintln!("merge_pattern: [");
        //for idx in tb::<_, [u32; 8]>(merge_pattern_idx) {
        //    eprintln!("    {idx:b},");
        //}
        //eprintln!("]");
        //let eq_mask = compare_with_next(board_mm256, interleaved);
        //let merge_pattern_idx = mask_to_idx(eq_mask);

        extract_board(merged)
    }
}

#[derive(Debug, Error)]
#[error("Required CPU features (AVX2 and SSSE3) are not available on this platform.")]
pub struct MissingCpuFeatures;

impl BoardAvx2 {
    /// Safely creates a `Board` from a 2D array.
    ///
    /// # Errors
    /// Returns a `MissingCpuFeatures` error if the CPU does not support the required AVX2 and SSSE3 features.
    pub fn from_array(cells: [[u8; 4]; 4]) -> Result<Self, MissingCpuFeatures> {
        if cfg!(target_arch = "x86_64")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("ssse3")
        {
            // Safety: We've verified the platform supports AVX2 and SSSE3.
            Ok(unsafe { Self::from_array_unchecked(cells) })
        } else {
            Err(MissingCpuFeatures)
        }
    }

    /// # Safety
    /// This function uses unsafe SIMD intrinsics. The caller must ensure that the
    /// target CPU supports AVX2 and SSSE3 instructions.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "ssse3")]
    pub unsafe fn from_array_unchecked(cells: [[u8; 4]; 4]) -> Self {
        unsafe {
            let flat_cells: [u8; 16] = std::mem::transmute(cells); // Flatten 2D array to 1D
            let board = _mm_loadu_si128(flat_cells.as_ptr() as *const __m128i);
            Self(set_msb(board))
        }
    }

    pub fn to_array(self) -> [[u8; 4]; 4] {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        // SAFETY: Board has the same bit representation as a byte slice
        unsafe { std::mem::transmute(remove_msb(self.0)) }
    }

    /// Compact rows of a 2048 board using SIMD intrinsics.
    pub fn swipe_right(self) -> Self {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        Self(unsafe { swipe_right_simd(self.0) })
    }
}

impl fmt::Debug for BoardAvx2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows = self.to_array().into_iter();

        if let Some(row) = rows.next() {
            row.iter().try_for_each(|c| write!(f, "{c:2x}"))?
        }

        for row in rows {
            f.write_char('\n')?;
            row.iter().try_for_each(|c| write!(f, "{c:2x}"))?
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::seq::{IndexedRandom, SliceRandom};

    use super::*;

    fn generate_random_board<const N: usize, const M: usize>(
        filled: u8,
        duplicates: u8,
    ) -> [[u8; N]; M] {
        let mut nums = Vec::with_capacity(16);
        nums.extend(1..filled + 1);

        // Add duplicates
        let duplicates = (0..duplicates)
            .map(|_| *nums.choose(&mut rand::rng()).unwrap())
            .collect_vec();

        nums.extend(duplicates);
        nums.resize(N * M, 0);

        // Shuffle the values randomly
        nums.shuffle(&mut rand::rng());
        let mut nums = nums.into_iter();

        use std::array as arr;
        arr::from_fn(|_| arr::from_fn(|_| nums.next().unwrap_or(0)))
    }

    fn baseline_swipe<const N: usize, const M: usize>(board: [[u8; N]; M]) -> [[u8; N]; M] {
        board.map(|mut row| {
            crate::swipe_right_u8_inf_arr(&mut row);
            row.reverse();
            row
        })
    }

    #[test]
    fn test_compact() {
        const N: i32 = 2000;

        let test_cases = (0..16).flat_map(|filled| {
            (0..N).map(move |_|
            // Generate a random board with the specified number of filled cells
            generate_random_board::<4, 4>(filled, 0))
        });

        for board in test_cases {
            let board_instance = BoardAvx2::from_array(board).unwrap();
            let baseline_output = baseline_swipe(board);
            let optimized_output = board_instance.swipe_right();
            let baseline_board = BoardAvx2::from_array(baseline_output).unwrap();

            assert_eq!(
                optimized_output.to_array(),
                baseline_output,
                "Mismatch found for board: \n{:?}\nBaseline:\n{:?}\nOptimized:\n{:?}",
                board_instance,
                baseline_board,
                optimized_output
            );
        }
    }

    #[test]
    fn test_merge() {
        let board = [[0, 1, 0, 1], [0, 2, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]];
        //let board = [[0, 0, 0, 1], [0, 0, 2, 3], [0, 3, 4, 5], [6, 7, 8, 9]];
        //let board = [[0, 0, 0, 1], [0, 0, 2, 2], [0, 3, 3, 3], [4, 4, 4, 4]];
        //let board = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4]];
        //let board = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];

        let board_instance = BoardAvx2::from_array(board).unwrap();
        let baseline_output = baseline_swipe(board);
        let optimized_output = board_instance.swipe_right();

        let baseline_board = BoardAvx2::from_array(baseline_output).unwrap();

        assert_eq!(
            optimized_output.to_array(),
            baseline_output,
            "Mismatch found for board: \n{:?}\nBaseline:\n{:?}\nOptimized:\n{:?}",
            board_instance,
            baseline_board,
            optimized_output
        );
    }

    #[test]
    #[ignore]
    fn test_endian() {
        let a = [[0u8, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [
            12, 13, 14, 15,
        ]];

        fn f<T: std::fmt::LowerHex>(inp: &[T]) {
            eprint!("merge_pattern: [");

            eprint!("{:#0x}", inp[0]);
            for idx in &inp[1..] {
                eprint!(", {idx:#0x}");
            }
            eprintln!("]");
        }

        unsafe {
            f(&tb::<_, [u8; 16]>(a));
            f(&tb::<_, [u16; 8]>(a));
            f(&tb::<_, [u32; 4]>(a));
            f(&tb::<_, [u64; 2]>(a));
            f(&tb::<_, [u128; 1]>(a));
            let b = _mm_loadu_si32(a.as_ptr().cast());
            let c = _mm_loadu_ps(a.as_ptr().cast());
        }

        let bytes = a.as_flattened().try_into().unwrap();
        let asdfa = u128::from_le_bytes(bytes);
        f(&[asdfa]);

        panic!();
    }
}
