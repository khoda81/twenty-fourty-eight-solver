fn main() -> anyhow::Result<()> {
    let inputs: anyhow::Result<Vec<_>> = std::io::stdin()
        .lines()
        .map(|line| Ok(line?.parse()?))
        .collect();

    swipe_vec(inputs?)
        .into_iter()
        .for_each(|out| println!("{out}"));

    Ok(())
}

#[inline(never)]
fn swipe_vec(mut slice: Vec<u16>) -> Vec<u16> {
    for b in slice.iter_mut() {
        *b = swipe_right_bitmask_u16(*b)
    }

    slice
}

#[inline(never)]
fn swipe_right_bitmask_u16(b: u16) -> u16 {
    twenty_fourty_eight_solver::swipe_right_bitmask_u16(b)
}
