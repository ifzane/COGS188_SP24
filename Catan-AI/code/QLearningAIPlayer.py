import numpy as np
from board import *
from player import *

class QLearningAIPlayer(player):
    
    def __init__(self, name, playerColor, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        super().__init__(name, playerColor)  # Initialize parent class attributes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.action_key_to_index = {}
        self.index_to_action_key = {}

    # This is currently taken from the heuristic so its random, need to fix
    def updateAI(self): 
        self.isAI = True
        self.setupResources = [] #List to keep track of setup resources
        #Initialize resources with just correct number needed for set up
        self.resources = {'ORE':0, 'BRICK':4, 'WHEAT':2, 'WOOD':4, 'SHEEP':2} #Dictionary that keeps track of resource amounts
        print("Added new AI Player:", self.name)
    
    def initial_setup(self, board):
        #Build random settlement
        possibleVertices = board.get_setup_settlements(self)

        #Simple heuristic for choosing initial spot
        diceRoll_expectation = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1, None:0}
        vertexValues = []

        #Get the adjacent hexes for each hex
        for v in possibleVertices.keys():
            vertexNumValue = 0
            resourcesAtVertex = []
            #For each adjacent hex get its value and overall resource diversity for that vertex
            for adjacentHex in board.boardGraph[v].adjacentHexList:
                resourceType = board.hexTileDict[adjacentHex].resource.type
                if(resourceType not in resourcesAtVertex):
                    resourcesAtVertex.append(resourceType)
                numValue = board.hexTileDict[adjacentHex].resource.num
                vertexNumValue += diceRoll_expectation[numValue] #Add to total value of this vertex

            #basic heuristic for resource diversity
            vertexNumValue += len(resourcesAtVertex)*2
            for r in resourcesAtVertex:
                if(r != 'DESERT' and r not in self.setupResources):
                    vertexNumValue += 2.5 #Every new resource gets a bonus
            
            vertexValues.append(vertexNumValue)


        vertexToBuild_index = vertexValues.index(max(vertexValues))
        vertexToBuild = list(possibleVertices.keys())[vertexToBuild_index]

        #Add to setup resources
        for adjacentHex in board.boardGraph[vertexToBuild].adjacentHexList:
            resourceType = board.hexTileDict[adjacentHex].resource.type
            if(resourceType not in self.setupResources and resourceType != 'DESERT'):
                self.setupResources.append(resourceType)

        self.build_settlement(vertexToBuild, board)


        #Build random road
        possibleRoads = board.get_setup_roads(self)
        randomEdge = np.random.randint(0, len(possibleRoads.keys()))
        self.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board)


    def update_q_table(self, state, action, reward, next_state, board):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.get_possible_actions(board)))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.get_possible_actions(board)))

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta


    def choose_action(self, state, board):
        possible_actions = self.get_possible_actions(board)
        action_keys = list(possible_actions.keys())
        
        self.update_action_mappings(action_keys)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.get_possible_actions(board)))
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.choice(len(action_keys))
        else:
            action_index = np.argmax(self.q_table[state])
        return action_index
        
    def update_action_mappings(self, action_keys):
        self.action_key_to_index = {key: index for index, key in enumerate(action_keys)}
        self.index_to_action_key = {index: key for key, index in self.action_key_to_index.items()}


    def get_state(self, board):
        state = (
            self.settlementsLeft,
            self.roadsLeft,
            self.citiesLeft,
            tuple(self.resources.values()),
            self.knightsPlayed,
            self.maxRoadLength,
            self.victoryPoints,
            self.longestRoadFlag,
            self.largestArmyFlag,
            #Add something about board state here
            )
        return state
    

    def get_possible_actions(self, board):
        possible_actions = {}
    
        # Check if the player has enough resources to build settlements
        if self.resources['BRICK'] >= 1 and self.resources['WOOD'] >= 1 and self.resources['SHEEP'] >= 1 and self.resources['WHEAT'] >= 1:
            possible_settlements = board.get_potential_settlements(self)
            settlement_commands = self.get_settlement_build_commands(possible_settlements)
            possible_actions.update(settlement_commands)
    
        # Check if the player has enough resources to build roads
        if self.resources['BRICK'] >= 1 and self.resources['WOOD'] >= 1:
            possible_roads = board.get_potential_roads(self)
            road_commands = self.get_road_build_commands(possible_roads)
            possible_actions.update(road_commands)
    
        # Check if the player has enough resources to build cities
        if self.resources['ORE'] >= 3 and self.resources['WHEAT'] >= 2:
            possible_cities = board.get_potential_cities(self)
            city_commands = self.get_city_build_commands(possible_cities)
            possible_actions.update(city_commands)

        possible_actions['end_turn'] = None

        return possible_actions
    

    def merge_command_dictionaries(self, commands):
        merged_dict = {}

        for command in commands:
            merged_dict.update(command)

        return merged_dict


    def get_road_build_commands(self, potential_roads):
        build_commands = {}
        command_index = 1
        for road in potential_roads.keys():
            build_commands[f'build_road_{command_index}'] = road
            command_index += 1
        return build_commands
    
    def get_settlement_build_commands(self, potential_settlements):
        build_commands = {}
        command_index = 1
        for settlement in potential_settlements.keys():
            build_commands[f'build_settlement_{command_index}'] = settlement
            command_index += 1
        return build_commands
    
    def get_city_build_commands(self, potential_cities):
        build_commands = {}
        command_index = 1
        for city in potential_cities.keys():
            build_commands[f'build_city_{command_index}'] = city
            command_index += 1
        return build_commands


    def calculate_reward(self):

        # Need to figure out a better reward value
        return self.victoryPoints


    def move(self, board):
        state = self.get_state(board)
        possible_actions = self.get_possible_actions(board)
        action_index = self.choose_action(state, board)
        action_key = self.index_to_action_key[action_index]
        action_value = possible_actions[action_key]
        print(f"Chosen action: {action_key} with value {action_value}")

        if action_key.startswith("build_settlement"):
            self.build_settlement(action_value, board)
            reward = self.calculate_reward()
        elif action_key.startswith("build_road"):
            self.build_road(action_value[0], action_value[1], board)
            reward = self.calculate_reward()
        elif action_key.startswith("build_city"):
            self.build_city(action_value, board)
            reward = self.calculate_reward()
        else:
            reward = 0 

        next_state = self.get_state(board)
        self.update_q_table(state, action_index, reward, next_state, board)

    def heuristic_move_robber(self, board):
        '''Function to control heuristic AI robber
        Calls the choose_player_to_rob and move_robber functions
        args: board object
        '''
        #Get the best hex and player to rob
        hex_i, playerRobbed = self.choose_player_to_rob(board)

        #Move the robber
        self.move_robber(hex_i, board, playerRobbed)

        return
    
    def choose_player_to_rob(self, board):
        '''Heuristic function to choose the player with maximum points.
        Choose hex with maximum other players, Avoid blocking own resource
        args: game board object
        returns: hex index and player to rob
        '''
        #Get list of robber spots
        robberHexDict = board.get_robber_spots()
        
        #Choose a hexTile with maximum adversary settlements
        maxHexScore = 0 #Keep only the best hex to rob
        for hex_ind, hexTile in robberHexDict.items():
            #Extract all 6 vertices of this hexTile
            vertexList = polygon_corners(board.flat, hexTile.hex)

            hexScore = 0 #Heuristic score for hexTile
            playerToRob_VP = 0
            playerToRob = None
            for vertex in vertexList:
                playerAtVertex = board.boardGraph[vertex].state['Player']
                if playerAtVertex == self:
                    hexScore -= self.victoryPoints
                elif playerAtVertex != None: #There is an adversary on this vertex
                    hexScore += playerAtVertex.visibleVictoryPoints
                    #Find strongest other player at this hex, provided player has resources
                    if playerAtVertex.visibleVictoryPoints >= playerToRob_VP and sum(playerAtVertex.resources.values()) > 0:
                        playerToRob_VP = playerAtVertex.visibleVictoryPoints
                        playerToRob = playerAtVertex
                else:
                    pass

            if hexScore >= maxHexScore and playerToRob != None:
                hexToRob_index = hex_ind
                playerToRob_hex = playerToRob
                maxHexScore = hexScore

        return hexToRob_index, playerToRob_hex