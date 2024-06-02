# Outline for Q learning player
import numpy as np
import random
import collections

class QLearningAgent():

    # This is the same as "player.py" except with the Q-learning specfic parameters
    def __init__(self, playerName, playerColor, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.name = playerName
        self.color = playerColor
        self.victoryPoints = 0
        self.isAI = True

        self.settlementsLeft = 5
        self.roadsLeft = 15
        self.citiesLeft = 4
        self.resources = {'ORE':5, 'BRICK':6, 'WHEAT':3, 'WOOD':6, 'SHEEP':3} 

        self.knightsPlayed = 0
        self.largestArmyFlag = False

        self.maxRoadLength = 0
        self.longestRoadFlag = False

        self.buildGraph = {'ROADS':[], 'SETTLEMENTS':[], 'CITIES':[]}
        self.portList = []

        self.newDevCards = []
        self.devCards = {'KNIGHT':0, 'VP':0, 'MONOPOLY':0, 'ROADBUILDER':0, 'YEAROFPLENTY':0}
        self.devCardPlayedThisTurn = False

        self.visibleVictoryPoints = self.victoryPoints - self.devCards['VP']

        # Q-learning specific parameters
        self.q_table = collections.defaultdict(lambda: np.zeros(self.action_space_size()))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def action_space_size(self):
        # Need to somehow figure out what actions you can take at any point
        return 4
    
    def get_state(self):
        # Seems pretty self explanatory
        pass

    def get_reward(self, board):
        # This seems like an okay way to start but maybe want to add more rewards
        return self.victoryPoints

    def get_best_action(self, state):
        possible_actions = self.get_possible_actions(state)
        if not possible_actions:
            return None
        return max(possible_actions, key=lambda action: self.q_table[state][action])
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
        else:
            return self.get_best_action(state)

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


    def move(self, board):
        state = self.get_state(board)
        action = self.choose_action(state)
        self.execute_action(action, board)
        next_state = self.get_state(board)
        reward = self.get_reward(board)
        self.update_q_table(state, action, reward, next_state)

    # 
    def execute_action(self, action, board):
        if action == 'build_settlement':
            pass
        elif action == 'build_city':
            pass
        elif action == 'build_road':
            pass
        elif action == 'trade':
            pass
        elif action == 'draw_dev_card':
            self.draw_devCard(board)
