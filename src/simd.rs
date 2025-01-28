use std::{
    arch::x86_64::*,
    fmt::{self, Write},
};

#[derive(Clone, Copy)]
pub struct Board(__m128i);

/// Load the lookup table as an AVX2 register.
unsafe fn load_lookup_table() -> __m256 {
    const E: i8 = -128;
    const PERM_LOOKUP: [[i8; 4]; 8] = [
        [3, E, E, E], // 000
        [0, 3, E, E], // 100
        [1, 3, E, E], // 010
        [0, 1, 3, E], // 110
        [2, 3, E, E], // 001
        [0, 2, 3, E], // 101
        [1, 2, 3, E], // 011
        [0, 1, 2, 3], // 111
    ];

    let lookup_ptr = PERM_LOOKUP.as_flattened().as_ptr();
    unsafe { _mm256_loadu_ps(lookup_ptr as *const f32) }
}

/// Interleave the board data for mask extraction.
unsafe fn interleave_board(board_mm128: __m128i) -> __m256i {
    unsafe {
        let board_mm256 = _mm256_set_m128i(board_mm128, board_mm128); // 01230123
        //let board_mm256d = _mm256_castsi256_pd(board_mm256);
        //
        ////let board_mm256 = _mm256_castsi128_si256(board_mm128); // 0123xxxx
        //
        ////_mm256_unpacklo_epi32(board_mm256, board_mm256)
        ////_mm256_shuffle_ps(board_mm256, board_mm256, 0b)
        //let zeros = _mm256_setzero_pd();
        //let intermediate = _mm256_shuffle_pd::<0b1100>(board_mm256d, zeros);
        //let swapped = _mm256_castpd_ps(intermediate);
        //
        //let result = _mm256_shuffle_ps::<0b10_01_11_00>(swapped, swapped);
        //_mm256_castps_si256(result)

        // 0x808080801f1e1d1c808080801b1a191880808080070605048080808003020100
        let indices = [
            0x80808080_03020100_u64,
            0x80808080_07060504_u64,
            0x80808080_1b1a1918_u64,
            0x80808080_1f1e1d1c_u64,
        ];

        // Load into a YMM register
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(board_mm256, indices)
    }
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
        let row_offsets = _mm_set_epi32(0x0c0c0c0c, 0x08080808, 0x04040404, 0x00000000);
        _mm_add_epi32(row_permutations, row_offsets)
    }
}

impl Board {
    pub fn from_array(cells: [[u8; 4]; 4]) -> Self {
        let flat_cells: [u8; 16] = unsafe { std::mem::transmute(cells) }; // Flatten 2D array to 1D
        unsafe {
            let board = _mm_loadu_si128(flat_cells.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();

            // Compare for non-zero elements
            let zero_mask = _mm_cmpeq_epi8(board, zero); // 0xFF for zeros
            //let nonzero_mask = _mm_andnot_si128(nonzero_mask, _mm_set1_epi8(-1)); // Invert the mask

            // Set MSB based on mask
            let msb = _mm_set1_epi8(-0x80); // MSB set in every byte
            let result = _mm_or_si128(_mm_andnot_si128(zero_mask, msb), board);

            Self(result)
        }
    }

    pub fn to_array(self) -> [[u8; 4]; 4] {
        // SAFETY: Board has the same bit representation as a byte slice
        unsafe {
            let zeros = _mm_setzero_si128();
            let m = _mm_set1_epi8(0x7f);
            let b = _mm_and_si128(self.0, m);
            let result = _mm_blendv_epi8(zeros, b, self.0);
            std::mem::transmute(result)
        }
    }

    // Compute non-zero mask using a single SIMD instruction approach
    unsafe fn nonzero(self) -> u16 {
        unsafe { _mm_movemask_epi8(self.0) as u16 }
    }

    /// Compact rows of a 2048 board using SIMD intrinsics.
    ///
    /// # Safety
    /// This function uses unsafe SIMD intrinsics. The caller must ensure that the
    /// target CPU supports AVX2 and SSSE3 instructions.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "ssse3")]
    pub unsafe fn compact_rows_unsafe(self) -> Self {
        let Self(board_mm128) = self;

        unsafe {
            let interleaved = interleave_board(board_mm128);

            let mask = _mm256_movemask_epi8(interleaved);
            let pattern_idx = mask_to_idx(mask);

            let lookup_mm256 = load_lookup_table();

            let row_permutations = get_row_permutations(lookup_mm256, pattern_idx);

            let permuted_rows = add_offsets(row_permutations);
            let result = _mm_shuffle_epi8(board_mm128, permuted_rows);
            Self(result)
        }
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
        use std::array as arr;
        let mut nums = nums.into_iter();

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
            // Create a Board instance from the array for your compact method
            let board_instance = Board::from_array(board);

            // Run the baseline swipe
            let baseline_output = baseline_swipe(board);

            // Run your optimized compact method
            let optimized_output = unsafe { board_instance.compact_rows_unsafe() };

            // Assert that the outputs are equal
            assert_eq!(
                optimized_output.to_array(),
                baseline_output,
                "Mismatch found for board: \n{:?}\nBaseline:\n{:?}\nOptimized:\n{:?}",
                Board::from_array(board),
                Board::from_array(baseline_output),
                optimized_output
            );
        }
    }
}
