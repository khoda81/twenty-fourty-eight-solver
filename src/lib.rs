#![feature(assert_matches)]
#![feature(slice_swap_unchecked)]

pub mod board;
pub mod search;

pub fn swipe_left_u8_inf_arr(cells: &mut [u8]) -> usize {
    // Find first non empty cell
    let Some(current) = cells
        .iter()
        .enumerate()
        .find_map(|(i, &c)| (c != 0).then_some(i))
    else {
        return 0;
    };

    let mut last = 0; // Write ptr
    cells.swap(last, current);

    for current in current + 1..cells.len() {
        if cells[current] == 0 {
            continue;
        } else if cells[current] == cells[last] {
            cells[last] += 1;
            cells[current] = 0;
            last += 1;
        } else {
            (cells[last] != 0).then(|| last += 1);
            cells.swap(last, current);
        }
    }

    last
}

pub fn swipe_right_u8_inf_arr(row: &mut [u8]) {
    row.reverse();
    swipe_left_u8_inf_arr(row);
    row.reverse();
}
