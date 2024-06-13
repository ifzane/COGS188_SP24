# COGS188_SP24

Welcome to the repository for the SP24 COGS 188 Final Project. We created a Q-learning agent that learned to play Catan and compared various hyperparameters in the learning process. If you want to run the code as given, you can just run the QLearningGame.py file.

## Group Members
- Tom Hocquet
- Ian Zane
- Kai Stern

## Catan Implementation

This was implemented using an adapted version of the Catan AI developed by Karan Vombatkere. The original is accessible at the link below:
[Karan Vombatkere's Catan AI](https://github.com/kvombatkere/Catan-AI)

## Files

### [Catan-AI](https://github.com/kpstern/COGS188_SP24/tree/main/Catan-AI)
Folder containing the edited versions of Karan Vombatkere's Catan-AI plus our modified or implemented files.

#### [pycache](https://github.com/kpstern/COGS188_SP24/tree/main/Catan-AI/__pycache__)
Folder containing various objects for use in the game.

#### [code](https://github.com/kpstern/COGS188_SP24/tree/main/Catan-AI/code)
Folder containing the code used in the project.

- [AIGame.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/AIGame.py): Implementation to play a game with just the heuristic AI developed by Vombatkere.
- [QLearningAIPlayer.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/QLearningAIPlayer.py): Our implementation of the agent using Q-learning.
- [QLearningGame.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/QLearningGame.py): Implementation to play a game with just Q-learning agents.
- [board.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/board.py): The class that manages the Catan game board.
- [catanGame.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/catanGame.py): Implementation to play a game with one heuristic AI and the rest humans.
- [gameView.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/gameView.py): Pygame display for the Catan game.
- [heuristicAIPlayer.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/heuristicAIPlayer.py): Original implementation of the heuristic AI.
- [player.py](https://github.com/kpstern/COGS188_SP24/blob/main/Catan-AI/code/player.py): Base player class which is expanded upon for AI players (either heuristic or Q-learning).

#### [images](https://github.com/kpstern/COGS188_SP24/tree/main/Catan-AI/images)
Example images of the Catan game implementation.

#### [old_q_tables](https://github.com/kpstern/COGS188_SP24/tree/main/Catan-AI/old_q_tables)
Examples of the format of the Q-tables learned by the agents.

### [Proposal_Project_TIK](https://github.com/kpstern/COGS188_SP24/blob/main/Proposal_Project_TIK.pdf)
Original project proposal.

### [FinalProject_TIK](https://github.com/kpstern/COGS188_SP24/blob/main/FinalProject_TIK.pdf)
Writeup of the final project.
