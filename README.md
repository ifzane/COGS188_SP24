This is the repository for the SP24 COGS 188 Final Project. We created a Q-learning agent that learned to play Catan and compared various hyperparameters in the learning process.

Group Members:

Tom Hocquet
Ian Zane
Kai Stern

Catan Implementation:

This was implemented using an adapted version of the Catan AI developed by Karan Vombatkere. The original is accessible at the link below
https://github.com/kvombatkere/Catan-AI


Files:

Catan-AI: Folder containing the edited versions of Karan Vombatkere's Catan-AI plus our modified or implemented files

  pycache: Folder containing various objects for use in the game
  
  code: Folder containing the code used in the project
  
    AIGame: Implementation to play a game with just the heuristic AI developed by Vombatkere
    
    QLearningAIPlayer: Our implementation of the agent using q-learning
    
    QLearningGame: Implementation to play a game with just q-learning agents
    
    board.py: The class that manages the catan game board
    
    catanGame: Implementation to play a game with one heuristic AI and the rest humans
    
    gameView: pygame display for the catan game
    
    heuristicAIPlayer: Original implementation of the heuristic AI
    
    player: base player class which is expanded upon for AI players (either heuristic or Q-learning)
    
  images: Example images of the catan game implementation
  
  old_q_tables: examples of the format of the q_tables learned by the agents
  
Proposal_Project_TIK: Original project proposal

FinalProject_TIK: Writeup of the final project
