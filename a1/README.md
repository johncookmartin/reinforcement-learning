# Instructions for running programs

All algorithms use classes and definitions from the `util` folder. The `util` folder must be in the same folder as the programs for them to run properly. Both the UCB and learning automata implementations are set to run the same seeded bandit by default.

## UCB

To run the implementation of the UCB algorithm use:

```bash
python3 ucb.py
```

By default this will run the implementation with:

- 10 arms for the bandit
- 100 trials
- 5000 rounds per trial
- Confidence rate of 1
- Seed 42

You are able to change any of these values through command line arguments. Run `python3 ucb.py --help` to see the arguments.

## Learning Automata

To run the implementation of the learning automata algorithm use:

```bash
python3 learning_automata.py
```

By default this will run the implementation with:

- 10 arms for the bandit
- 100 trials
- 5000 rounds per trial
- Reward rate of 0.01
- Penalty rate of 0.01 (where applicable)

You are able to change any of these values through command line arguments. Run `python3 learning_automata.py --help` to see the arguments.

## Tic Tac Toe

To run the implementation of the tic tac toe game use:

```bash
python3 tic_tac_toe.py
```

This will create a player and opponent for each of the various opponent types:

- **RANDOM**: Opponent will play random moves
- **RANDOM_ROW**: Opponent will attempt to play randomly in the same row that the player just played or will play a random move
- **RANDOM_COL**: Opponent will attempt to play randomly in the same column that the player just played or will play a random move
- **RANDOM_DIAG**: Opponent will attempt to play randomly in the same diagonal as the player just played
- **MIRROR**: Opponent will play a move opposite from what the player played
- **SELF**: Opponent will employ the same learning strategy as the player

For each of these match ups, one trial is played with a player that plays only greedy moves and one trial is played with a player that employs a greedy epsilon approach. Each trial consists of a maximum of 10000 games and will end earlier if the average change in state value evaluation converges below 0.001.

By default the program runs with:

- 10000 games per trial
- Seed 42
- Learning rate of 0.1
- Exploring rate of 0.1

These can be changed as command line arguments. Enter `python3 tic_tac_toe.py --help` to see the arguments.

### Note

The tic tac toe game was implemented using a 9-bit and 18-bit integer tracking system. A description of how the board is being represented this way can be found in [util/tic_tac_toe_game.py](util/tic_tac_toe_game.py).

## Troubleshooting

There are a number of plots that use matplotlib. These shouldn't run on the moon server and the function has been set up to avoid attempting to run the plots if matplotlib is not detected. However, if there are problems, first try adding a return statement after line 30 of [util/functions.py](util/functions.py) to prevent any plots from running.
