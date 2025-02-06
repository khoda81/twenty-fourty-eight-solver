use std::{
    arch::x86_64::{__m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8},
    fmt::{self, Write},
};

use thiserror::Error;

pub mod editor;
pub mod simd_utils;

#[derive(Clone, Copy)]
pub struct BoardAvx2(pub(crate) __m128i);

#[derive(Debug, Error)]
#[error("Required CPU features (AVX2 and SSSE3) are not available on this platform.")]
pub struct MissingCpuFeatures;

impl BoardAvx2 {
    /// Safely creates a `Board` from a 2D array.
    ///
    /// # Errors
    /// Returns a `MissingCpuFeatures` error if the CPU does not support the required AVX2 and SSSE3 features.
    pub fn new() -> Result<Self, MissingCpuFeatures> {
        if cfg!(target_arch = "x86_64")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("ssse3")
        {
            // Safety: We've verified the platform supports AVX2 and SSSE3.
            Ok(unsafe { Self::from_array_unchecked([[0; 4]; 4]) })
        } else {
            Err(MissingCpuFeatures)
        }
    }

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
            Self(simd_utils::set_msb(board))
        }
    }

    pub fn to_array(self) -> [[u8; 4]; 4] {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        // SAFETY: Board has the same bit representation as a byte slice
        unsafe { std::mem::transmute(simd_utils::remove_msb(self.0)) }
    }

    /// Compact rows of a 2048 board using SIMD intrinsics.
    #[inline]
    pub fn swipe_right(self) -> Self {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        Self(unsafe { simd_utils::swipe_right_simd(self.0) })
    }

    pub fn swipe_direction(mut self, direction: u16) -> BoardAvx2 {
        for move_idx in 0..4 {
            if direction == move_idx {
                self = self.swipe_right();
            }

            self = self.rotate_90();
        }

        self
    }

    /// Compact rows of a 2048 board using SIMD intrinsics.
    #[inline]
    pub fn checked_swipe_right(self) -> Option<Self> {
        // SAFETY: Board is only instantiatable on avx2 ssse3
        let new_self = Self(unsafe { simd_utils::swipe_right_simd(self.0) });
        (self != new_self).then_some(new_self)
    }

    /// Rotate the board 90deg
    #[inline]
    pub fn rotate_90(self) -> Self {
        Self(unsafe { simd_utils::rotate_90(self.0) })
    }

    pub fn into_inner(self) -> __m128i {
        self.0
    }

    pub fn into_u128(self) -> u128 {
        unsafe { std::mem::transmute(self) }
    }

    pub fn num_empty(self) -> u32 {
        let msb = unsafe { _mm_movemask_epi8(self.0) as u16 };
        msb.count_zeros()
    }
}

impl PartialEq for BoardAvx2 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, other.0)) == 0xFFFF }
    }
}

impl Eq for BoardAvx2 {}

impl PartialOrd for BoardAvx2 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BoardAvx2 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::mem::transmute;

        let a: u128 = unsafe { transmute(self.0) };
        let b: u128 = unsafe { transmute(other.0) };

        a.cmp(&b)
    }
}

fn val_to_char(v: u8) -> char {
    match v {
        0 => '.',
        1..=9 => (b'0' + v) as char,
        a if a & 0b01100000 != 0 => match a & 0b1111 {
            1 => 'o',
            2 => 't',
            a => (a + b'A') as char,
        },
        a => (a - 10 + b'a') as char,
    }
}

fn format_row<'a>(
    f: &mut fmt::Formatter<'_>,
    row: impl IntoIterator<Item = &'a u8>,
) -> Result<(), fmt::Error> {
    let mut row = row.into_iter();

    if let Some(c) = row.next() {
        f.write_char(val_to_char(*c))?;
    }

    row.try_for_each(|c| {
        f.write_char(' ')?;
        f.write_char(val_to_char(*c))
    })
}

impl fmt::Debug for BoardAvx2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows = self.to_array().into_iter();

        if let Some(row) = rows.next() {
            format_row(f, &row)?
        }

        for row in rows {
            f.write_str(" | ")?;
            format_row(f, &row)?
        }

        Ok(())
    }
}

impl fmt::Display for BoardAvx2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows = self.to_array().into_iter();

        if let Some(row) = rows.next() {
            //format_row_display(f, &row)?
            format_row(f, &row)?
        }

        for row in rows {
            f.write_char('\n')?;
            //format_row_display(f, &row)?
            format_row(f, &row)?
        }

        Ok(())
    }
}

pub mod test_utils {
    use itertools::Itertools as _;
    use rand;
    use rand::seq::{IndexedRandom as _, SliceRandom};

    pub fn generate_random_board<const N: usize, const M: usize>(
        filled: u8,
        duplicates: u8,
    ) -> [[u8; N]; M] {
        let mut nums = Vec::with_capacity(16);
        nums.extend(1..filled + 1);

        // Add duplicates
        if !nums.is_empty() {
            let duplicates = (0..duplicates)
                .map(|_| *nums.choose(&mut rand::rng()).unwrap())
                .collect_vec();

            nums.extend(duplicates);
        }

        nums.resize(N * M, 0);

        // Shuffle the values randomly
        nums.shuffle(&mut rand::rng());
        let mut nums = nums.into_iter();

        use std::array as arr;
        arr::from_fn(|_| arr::from_fn(|_| nums.next().unwrap_or(0)))
    }

    pub fn baseline_swipe<const N: usize, const M: usize>(board: [[u8; N]; M]) -> [[u8; N]; M] {
        board.map(|mut row| {
            crate::swipe_left_u8_inf_arr(&mut row);
            //crate::swipe_right_u8_inf_arr(&mut row);
            //row.reverse();
            row
        })
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_compact() {
        const N: i32 = 200;

        let test_cases = (0..16).flat_map(|filled| {
            (0..N).map(move |_|
            // Generate a random board with the specified number of filled cells
            test_utils::generate_random_board::<4, 4>(filled, 0))
        });

        for board in test_cases {
            test_swipe(board)
        }
    }

    #[test]
    fn test_merge() {
        const N: i32 = 20;

        let test_cases = (0..16).flat_map(|filled| {
            (0..filled).cartesian_product(0..N).map(move |(dup, _)|
            // Generate a random board with the specified number of filled cells
            test_utils::generate_random_board::<4, 4>(filled, dup))
        });

        for board in test_cases {
            test_swipe(board)
        }
    }

    #[test]
    fn test_merge_0() {
        test_swipe([[0, 1, 0, 1], [0, 2, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]]);
    }

    #[test]
    fn test_merge_1() {
        test_swipe([[0, 0, 0, 1], [0, 0, 2, 3], [0, 3, 4, 5], [6, 7, 8, 9]]);
    }

    #[test]
    fn test_merge_2() {
        test_swipe([[0, 0, 0, 1], [0, 0, 2, 2], [0, 3, 3, 3], [4, 4, 4, 4]]);
    }

    #[test]
    fn test_merge_3() {
        test_swipe([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4]]);
    }

    #[test]
    fn test_merge_4() {
        test_swipe([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [
            13, 14, 15, 16,
        ]]);
    }

    #[test]
    fn test_merge_5() {
        test_swipe([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]);
    }

    #[test]
    fn test_rot90() {
        let board = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
        let output = [[12, 8, 4, 0], [13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3]];

        let board = BoardAvx2::from_array(board).unwrap();
        let rotated = board.rotate_90();
        let output = BoardAvx2::from_array(output).unwrap();
        assert_eq!(rotated, output, "Input: {board:?}",);
    }

    fn test_swipe(board: [[u8; 4]; 4]) {
        let board_instance = BoardAvx2::from_array(board).unwrap();
        let baseline_output = test_utils::baseline_swipe(board);
        let optimized_output = board_instance.swipe_right();

        let baseline_board = BoardAvx2::from_array(baseline_output).unwrap();

        assert_eq!(
            optimized_output, baseline_board,
            "Mismatch found for board: \n{:?}\nBaseline:\n{:?}\nOptimized:\n{:?}",
            board_instance, baseline_board, optimized_output
        );
    }
}
