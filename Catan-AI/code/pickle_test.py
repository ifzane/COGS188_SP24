import numpy as np
from board import *
from player import *
import pickle
import os
from QLearningGame import catanAIGame

num_episodes = int(input("Enter Number of Episodes"))
    
for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}/{num_episodes}")

    game_env = catanAIGame()

    # Create an instance of the QLearningAIPlayer
    #tom = QLearningAIPlayer(name='tom', playerColor='red')
    # ai_player2 = QLearningAIPlayer(name='AI_Player', playerColor='blue')
    # ai_player3 = QLearningAIPlayer(name='AI_Player', playerColor='green')

    # Load Q-values before starting the training
    #tom.load_q_values(q_values_file)
    # ai_player2.load_q_values(q_values_file)
    # ai_player3.load_q_values(q_values_file)
    
    # Reset the game environment for each new episode
    # game_env.__init__()
    
    # # Set AI player to the environment's player queue
    # #tom.updateAI()
    # # ai_player2.updateAI()
    # # ai_player3.updateAI()
    # #game_env.playerQueue.put(tom)
    # # game_env.playerQueue.put(ai_player2)
    # # game_env.playerQueue.put(ai_player3)
    
    # done = False
    # while not done:
    #     for currPlayer in game_env.playerQueue.queue:
    #         #if currPlayer == ai_player:
    #         currPlayer.move(game_env.board, game_env)
            

            # # Check if game is over
            # if currPlayer.victoryPoints >= game_env.maxPoints:
            #     done = True
            #     break
    
#     # Save Q-values after each episode
#     print(f"Episode {episode + 1} completed. Q-values saved to {q_values_file}")

# print("Training completed. Q-values saved to", q_values_file)