use std::{
    arch::x86_64::*,
    fmt::{self, Write},
};

use thiserror::Error;

#[derive(Clone, Copy)]
pub struct Board(__m128i);

/// Load the lookup table as an AVX2 register.
unsafe fn load_fill_table() -> __m256 {
    const E: i8 = -0x80; // Empty
    const PERM_LOOKUP: [[i8; 4]; 8] = [
        [3, E, E, E], // 000x
        [0, 3, E, E], // 100x
        [1, 3, E, E], // 010x
        [0, 1, 3, E], // 110x
        [2, 3, E, E], // 001x
        [0, 2, 3, E], // 101x
        [1, 2, 3, E], // 011x
        [0, 1, 2, 3], // 111x
    ];

    let lookup_ptr = PERM_LOOKUP.as_flattened().as_ptr();
    unsafe { _mm256_loadu_ps(lookup_ptr as *const f32) }
}

/// Load the lookup table as an AVX2 register.
unsafe fn load_merge_table() -> __m256 {
    const O: i8 = 0x81_u8 as i8; // Cell[1] + 1
    const T: i8 = 0x82_u8 as i8; // Cell[2] + 1
    const E: i8 = 0x83_u8 as i8; // Empty
    const PERM_LOOKUP: [[i8; 4]; 8] = [
        [0, 1, 2, 3], // 000x
        [O, 2, 3, E], // 100x
        [0, O, 3, E], // 010x
        [O, 2, 3, E], // 110x
        [0, 1, T, E], // 001x
        [O, T, E, E], // 101x
        [0, O, 3, E], // 011x
        [O, T, E, E], // 111x
    ];

    let lookup_ptr = PERM_LOOKUP.as_flattened().as_ptr();
    unsafe { _mm256_loadu_ps(lookup_ptr as *const f32) }
}

/// Interleave the board data for mask extraction.
unsafe fn interleave_board(board_mm256: __m256i) -> __m256i {
    let indices = [
        0xFFFFFFFF_03020100_u64,
        0xFFFFFFFF_07060504_u64,
        0xFFFFFFFF_1B1A1918_u64,
        0xFFFFFFFF_1F1E1D1C_u64,
    ];

    unsafe {
        // Load into a YMM register
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(board_mm256, indices)
    }
}

// Repeat the board twice in a ymm
unsafe fn broadcast_256(board_mm128: __m128i) -> __m256i {
    unsafe { _mm256_set_m128i(board_mm128, board_mm128) }
}

/// Broadcast the mask into an AVX2 register.
unsafe fn mask_to_idx(mask: i32) -> __m256i {
    unsafe {
        let mask_broadcast = _mm256_set1_epi32(mask);
        let mask_broadcast = _mm256_unpacklo_epi8(mask_broadcast, mask_broadcast);
        _mm256_unpacklo_epi8(mask_broadcast, mask_broadcast)
    }
}

/// Get row permutations based on the broadcasted mask.
unsafe fn get_row_permutations(lookup_mm256: __m256, mask_broadcast: __m256i) -> __m128i {
    unsafe {
        let row_permutations = _mm256_permutevar8x32_ps(lookup_mm256, mask_broadcast);
        let row_permutations = _mm256_castps256_ps128(row_permutations);
        _mm_castps_si128(row_permutations)
    }
}

/// Add row offsets to the row permutations.
unsafe fn add_offsets(row_permutations: __m128i) -> __m128i {
    unsafe {
        let row_offsets = _mm_set_epi32(0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000);
        _mm_add_epi32(row_permutations, row_offsets)
    }
}

/// Shift cells to left for comparison mask
unsafe fn comparison_target(board_mm256: __m256i) -> __m256i {
    let indices = [
        0xFFFFFFFF_FF030201_u64,
        0xFFFFFFFF_FF070605_u64,
        0xFFFFFFFF_FF1B1A19_u64,
        0xFFFFFFFF_FF1F1E1D_u64,
    ];

    unsafe {
        // Load into a YMM register
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(board_mm256, indices)
    }
}

/// Generate comparison mask
unsafe fn compare_with_next(board_mm256: __m256i, interleaved: __m256i) -> i32 {
    unsafe {
        let target = comparison_target(board_mm256);
        let mask_256 = _mm256_cmpeq_epi8(interleaved, target);

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
unsafe fn swipe_left_simd(board_mm128: __m128i) -> __m128i {
    unsafe {
        let board_mm256 = broadcast_256(board_mm128);
        let interleaved = interleave_board(board_mm256);

        let fill_mask = _mm256_movemask_epi8(interleaved);
        let fill_pattern_idx = mask_to_idx(fill_mask);

        let eq_mask = compare_with_next(board_mm256, interleaved);
        let merge_pattern_idx = mask_to_idx(eq_mask);

        let lookup_mm256 = load_fill_table();
        let row_permutations = get_row_permutations(lookup_mm256, fill_pattern_idx);
        let permuted_rows = add_offsets(row_permutations);
        _mm_shuffle_epi8(board_mm128, permuted_rows)
    }
}

#[derive(Debug, Error)]
#[error("Required CPU features (AVX2 and SSSE3) are not available on this platform.")]
pub struct MissingCpuFeatures;

impl Board {
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
            let zero = _mm_setzero_si128();

            // Compare for non-zero elements
            let zero_mask = _mm_cmpeq_epi8(board, zero); // 0xFF for zeros

            // Set MSB based on mask
            let msb = _mm_set1_epi8(0x80_u8 as i8); // MSB set in every byte
            let result = _mm_or_si128(_mm_andnot_si128(zero_mask, msb), board);

            Self(result)
        }
    }

    pub fn to_array(self) -> [[u8; 4]; 4] {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        // SAFETY: Board has the same bit representation as a byte slice
        unsafe { std::mem::transmute(remove_msb(self.0)) }
    }

    /// Compact rows of a 2048 board using SIMD intrinsics.
    pub fn swipe_left(self) -> Self {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        Self(unsafe { swipe_left_simd(self.0) })
    }
}

impl fmt::Debug for Board {
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

    use crate::swipe_left_u8_inf_arr;

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
            swipe_left_u8_inf_arr(&mut row);
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
            let board_instance = Board::from_array(board).unwrap();
            let baseline_output = baseline_swipe(board);
            let optimized_output = board_instance.swipe_left();
            let baseline_board = Board::from_array(baseline_output).unwrap();

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
        //let board = [[0, 1, 0, 1], [0, 2, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]];
        let board = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4]];

        let board_instance = Board::from_array(board).unwrap();
        let baseline_output = baseline_swipe(board);
        let board_mm128 = board_instance.0;

        let swiped_mm128 = unsafe {
            let board_mm256 = broadcast_256(board_mm128);
            let interleaved = interleave_board(board_mm256);

            eprintln!("interleaved: {:?}", tb::<_, [u32; 8]>(interleaved));
            let fill_mask = _mm256_movemask_epi8(interleaved);
            let fill_pattern_idx = mask_to_idx(fill_mask);

            let eq_mask = compare_with_next(board_mm256, interleaved);
            let merge_pattern_idx = mask_to_idx(eq_mask);
            eprintln!("eq_mask: {eq_mask:032b}");

            use std::mem::transmute as tb;
            eprintln!("merge_pattern: {:?}", tb::<_, [u32; 8]>(merge_pattern_idx));
            eprintln!(
                "merge_pattern: {:?}",
                tb::<_, [u32; 8]>(merge_pattern_idx).map(|i| i & 7)
            );

            eprintln!("merge_pattern: [");
            for idx in tb::<_, [u32; 8]>(merge_pattern_idx) {
                eprintln!("    {idx:b},");
            }
            eprintln!("]");

            let lookup_mm256 = load_fill_table();
            let row_permutations = get_row_permutations(lookup_mm256, fill_pattern_idx);
            let permuted_rows = add_offsets(row_permutations);
            _mm_shuffle_epi8(board_mm128, permuted_rows)
        };

        let optimized_output = Board(swiped_mm128);

        let baseline_board = Board::from_array(baseline_output).unwrap();

        assert_eq!(
            optimized_output.to_array(),
            baseline_output,
            "Mismatch found for board: \n{:?}\nBaseline:\n{:?}\nOptimized:\n{:?}",
            board_instance,
            baseline_board,
            optimized_output
        );
        panic!();
    }
}
