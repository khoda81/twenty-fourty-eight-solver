# 2048 Solver

This project provides a solver for the popular [2048 game](https://play2048.co/) using various search algorithms. The solver is implemented in Rust and includes support for MeanMax and Monte Carlo Tree Search (MCTS) algorithms. Additionally, it includes a parameter search script using Optuna for hyperparameter optimization.

## Features

- **MeanMax Algorithm**: A flexible search algorithm with depth and deadline constraints.
- **Monte Carlo Tree Search (MCTS)**: A probabilistic search algorithm with exploration rate customization.
- **Interactive Board Editor**: Allows users to set up custom board configurations.
- **Parameter Search**: Uses Optuna for hyperparameter optimization.

## Installation

To build and run the project, you need to have Rust installed. You can install Rust using [rustup](https://rustup.rs/).

```sh
# Clone the repository
git clone https://github.com/yourusername/twenty-fourty-eight-solver.git
cd twenty-fourty-eight-solver

# Build the project
cargo build --release
```

## Usage

### Command Line Interface

The solver can be run from the command line with various options:

```sh
# Run the solver with default settings
cargo run --release -- --mode single-game --algorithm mcts --search-time 0.05

# Run the solver with custom depth and search time
cargo run --release -- --mode single-game --algorithm meanmax --depth 3 --search-time 2.0
```

### Interactive Board Editor

To run the solver from a given starting position you can use the terminal editor.

1. **Run the Program**: Start the program by running the main executable with the `--board-editor` flag.

    ```sh
    cargo run --release -- --board-editor
    ```

2. **Edit the Board**:
    - Use the arrow keys (<kbd>Up</kbd>, <kbd>Down</kbd>, <kbd>Left</kbd>, <kbd>Right</kbd>) to move the cursor around the grid.
    - Use the <kbd>+</kbd> or <kbd>=</kbd> keys to increment the value of the current cell.
    - Use the <kbd>-</kbd> key to decrement the value of the current cell.
    - Use the number keys (<kbd>0</kbd>-<kbd>9</kbd>) or letter keys (<kbd>a</kbd>-<kbd>i</kbd>) to directly set the value of the current cell.
    - Use the <kbd>.</kbd> key to clear the current cell.
    - Press <kbd>Enter</kbd> or <kbd>q</kbd> to finish editing and exit the board editor.

3. **Save and Use the Board**: Once you exit the board editor, the board configuration you created will be used for the game or evaluation, depending on the mode you selected.

### Parameter Search

The parameter search script uses Optuna to find the best hyperparameters for the solver. Ensure you have Python and the required dependencies installed.

```sh
# Run the parameter search after building with cargo
python parameter-search.py
```
