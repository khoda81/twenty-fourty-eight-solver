use std::arch::x86_64::*;
use std::mem::transmute as tb;

#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! debug_print {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprint!($($arg)*);
        }
    };
}

#[allow(unused_variables)]
pub(crate) fn f<T: std::fmt::LowerHex>(name: &str, inp: &[T]) {
    debug_print!("{name:15}: [");

    debug_print!("{:#010x}", inp[0]);
    for idx in &inp[1..] {
        debug_print!(", {idx:#010x}");
    }

    debug_println!("]");
}

#[allow(unused_variables)]
pub(crate) fn fb<T: std::fmt::Binary>(name: &str, inp: &[T]) {
    debug_print!("{name:15}: [");

    debug_print!("{:#010b}", inp[0]);
    for idx in &inp[1..] {
        debug_print!(", {idx:#010b}")
    }

    debug_println!("]");
}

/// Load the lookup table as an AVX2 register.
pub(crate) unsafe fn load_fill_table() -> __m256 {
    const E: i8 = -0x80; // Empty
    const PERM_LOOKUP: [[i8; 4]; 8] = [
        [3, E, E, E], // x000
        [0, 3, E, E], // x001
        [1, 3, E, E], // x010
        [0, 1, 3, E], // x011
        [2, 3, E, E], // x100
        [0, 2, 3, E], // x101
        [1, 2, 3, E], // x110
        [0, 1, 2, 3], // x111
    ];

    let lookup_ptr = PERM_LOOKUP.as_flattened().as_ptr();
    unsafe { _mm256_loadu_ps(lookup_ptr as *const f32) }
}

/// Load the lookup table as an AVX2 register.
pub(crate) unsafe fn load_merge_table() -> __m256 {
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
pub(crate) unsafe fn interleave_board(board_mm256: __m256i) -> __m256i {
    let indices = [
        0xFFFFFFFF_03020100_u64,
        0xFFFFFFFF_07060504_u64,
        0xFFFFFFFF_0B0A0908_u64,
        0xFFFFFFFF_0F0E0D0C_u64,
    ];

    unsafe {
        // Load into a YMM register
        let indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
        _mm256_shuffle_epi8(board_mm256, indices)
    }
}

/// Broadcast the mask into an AVX2 register.
pub(crate) unsafe fn mask_to_idx(mask: i32) -> __m256i {
    let [d, c, b, a] = mask.to_le_bytes().map(|i| i as i8);

    unsafe {
        _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, a, -1, -1, -1, -1, -1, -1, -1, b, -1, -1, -1, -1, -1, -1,
            -1, c, -1, -1, -1, -1, -1, -1, -1, d,
        )
    }
}

pub(crate) unsafe fn merge_mask_to_idx(mask: i32) -> __m256i {
    let [d, c, b, a] = mask.to_le_bytes().map(|i| i as i8);

    unsafe {
        _mm256_set_epi8(
            -1, -1, -1, a, -1, -1, -1, b, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, c, -1, -1, -1, d,
        )
    }
}

/// Broadcast the mask into an AVX2 register.
pub(crate) unsafe fn extract_board(compacted_board: __m256i) -> __m128i {
    unsafe {
        let lo = _mm256_castsi256_si128(compacted_board); // Lower 128 bits (indices [0-3])
        let hi = _mm256_extracti128_si256::<1>(compacted_board); // Upper 128 bits (indices [4-7])

        // Create a shuffle mask for gathering [0, 2, 3] from lo and [5] from hi
        _mm_blend_epi32::<0b1100>(lo, hi)
    }
}

/// Broadcast the mask into an AVX2 register.
pub(crate) unsafe fn get_merge_target(compacted_board: __m256i) -> __m256i {
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
pub(crate) unsafe fn lookup(lookup_mm256: __m256, mask_broadcast: __m256i) -> __m256 {
    unsafe { _mm256_permutevar8x32_ps(lookup_mm256, mask_broadcast) }
}

/// Add row offsets to the row permutations.
pub(crate) unsafe fn add_offsets(row_permutations: __m256) -> __m256i {
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
pub(crate) unsafe fn add_merge_offsets(row_permutations: __m256) -> __m256i {
    unsafe {
        let row_offsets = _mm256_set_epi64x(
            0x0808080800000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0808080800000000,
        );

        let row_permutations = _mm256_castps_si256(row_permutations);
        _mm256_add_epi32(row_permutations, row_offsets)
    }
}

/// Shift cells to right for comparison mask
pub(crate) unsafe fn comparison_target(board_mm256: __m256i) -> __m256i {
    unsafe { _mm256_alignr_epi8::<1>(board_mm256, board_mm256) }
}

/// Generate comparison mask
pub(crate) unsafe fn compare_with_next(compacted_board_mm256: __m256i) -> i32 {
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
pub(crate) unsafe fn remove_msb(cells: __m128i) -> __m128i {
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
pub(crate) unsafe fn set_msb(board: __m128i) -> __m128i {
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
pub(crate) unsafe fn swipe_right_simd(board_mm128: __m128i) -> __m128i {
    #[allow(clippy::missing_transmute_annotations)]
    unsafe {
        f("input", &tb::<_, [u32; 4]>(board_mm128));
        //let fill_mask = _mm_movemask_epi8(board_mm128);
        //let fill_mask = fill_mask | ((fill_mask >> 4) << 16);

        let board_mm256 = _mm256_set_m128i(board_mm128, board_mm128); // u8x4x4x2
        let interleaved = interleave_board(board_mm256); // x x x x 0 1 2 3 x x x ...

        f("interleaved", &tb::<_, [u32; 8]>(interleaved));
        debug_println!("interleaved: {:?}", tb::<_, [u32; 8]>(interleaved));
        let fill_mask = _mm256_movemask_epi8(interleaved);
        let fill_pattern_idx = mask_to_idx(fill_mask); // u32x4

        debug_println!("fill_mask:       {:032b}", fill_mask);
        fb("fill_pattern_id", &tb::<_, [u32; 8]>(fill_pattern_idx));
        //debug_println!(
        //    "fill_pattern_idx: {:?}",
        //    tb::<_, [u32; 8]>(interleaved).map(|i| i & 7)
        //);

        let lookup_mm256 = load_fill_table();
        let row_permutations = lookup(lookup_mm256, fill_pattern_idx);
        let permutations = add_offsets(row_permutations);
        let compacted_board = _mm256_shuffle_epi8(board_mm256, permutations);

        f("compacted_board", &tb::<_, [u32; 8]>(compacted_board));

        let eq_mask = compare_with_next(compacted_board);
        let merge_pattern_idx = merge_mask_to_idx(eq_mask);
        debug_println!("eq_mask: {eq_mask:032b}");
        fb("eq_mask", &tb::<_, [u32; 8]>(merge_pattern_idx));
        let merge_lookup = load_merge_table();
        let merge_pattern = lookup(merge_lookup, merge_pattern_idx);
        f("merge_pattern", &tb::<_, [u32; 8]>(merge_pattern));
        let merge_target_idx = add_merge_offsets(merge_pattern);
        f("merge_target_id", &tb::<_, [u32; 8]>(merge_target_idx));

        let merge_target = get_merge_target(compacted_board);
        f("merge_target", &tb::<_, [u32; 8]>(merge_target));
        let merged = _mm256_shuffle_epi8(merge_target, merge_target_idx);
        f("merged", &tb::<_, [u32; 8]>(merged));

        debug_println!("merge_pattern: {:?}", tb::<_, [u32; 8]>(merge_pattern_idx));
        debug_println!(
            "merge_pattern: {:?}",
            tb::<_, [u32; 8]>(merge_pattern_idx).map(|i| i & 7)
        );

        fb("merge_pattern", &tb::<_, [u32; 8]>(merge_pattern_idx));

        extract_board(merged)
    }
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn rotate_90(board: __m128i) -> __m128i {
    unsafe {
        // Shuffle mask for 90-degree clockwise rotation
        let shuffle_mask = _mm_set_epi8(
            15, 11, 7, 3, // Last column becomes first row
            14, 10, 6, 2, // Third column becomes second row
            13, 9, 5, 1, // Second column becomes third row
            12, 8, 4, 0, // First column becomes fourth row
        );

        _mm_shuffle_epi8(board, shuffle_mask)
    }
}
