use crossterm::{
    QueueableCommand,
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal::{
        Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode,
    },
};
use std::io::{self, Write};

pub fn grid_editor() -> io::Result<[[u8; 4]; 4]> {
    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, Hide)?;

    let mut grid = [0u8; 16];
    let mut cursor = 0;

    loop {
        // Draw grid
        stdout.queue(Clear(ClearType::All))?;
        stdout.queue(MoveTo(0, 0))?;
        cursor %= 16;

        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                let val = grid[idx];

                let ch = super::val_to_char(val);
                let color = get_color(val);

                if idx == cursor {
                    stdout.queue(SetBackgroundColor(Color::Grey))?;
                } else {
                    stdout.queue(SetBackgroundColor(Color::Reset))?;
                }

                stdout
                    .queue(SetForegroundColor(color))?
                    .queue(Print(ch))?
                    .queue(SetBackgroundColor(Color::Reset))?
                    .queue(Print(' '))?
                    .queue(ResetColor)?;
            }

            stdout.queue(Print("\r\n"))?;
        }

        stdout.flush()?;

        // Handle input
        let event = event::read()?;
        if let Event::Key(KeyEvent {
            code,
            kind: KeyEventKind::Press,
            ..
        }) = event
        {
            match code {
                KeyCode::Enter => break,
                KeyCode::Char(c) => match c {
                    '+' => grid[cursor] = (grid[cursor] + 1).min(18),
                    '-' => grid[cursor] = grid[cursor].saturating_sub(1),
                    'q' => break,

                    '0'..='9' => {
                        grid[cursor] = c.to_digit(10).map(|n| n as u8).unwrap_or(grid[cursor]);
                        cursor += 1;
                    }

                    'a'..='i' => {
                        grid[cursor] = 10 + (c.to_ascii_lowercase() as u8 - b'a');
                        cursor += 1;
                    }

                    _ => {}
                },
                KeyCode::Up => cursor = cursor.wrapping_sub(4),
                KeyCode::Down => cursor = cursor.wrapping_add(4),
                KeyCode::Left => cursor = cursor.wrapping_sub(1),
                KeyCode::Right => cursor = cursor.wrapping_add(1),
                _ => {}
            }
        }
    }

    execute!(stdout, LeaveAlternateScreen, Show)?;
    disable_raw_mode()?;

    Ok([
        grid[0..4].try_into().unwrap(),
        grid[4..8].try_into().unwrap(),
        grid[8..12].try_into().unwrap(),
        grid[12..16].try_into().unwrap(),
    ])
}

fn get_color(v: u8) -> Color {
    match v {
        0 => Color::DarkGrey,
        1..=9 => Color::White,
        10..=18 => Color::Yellow,
        _ => Color::Reset,
    }
}
