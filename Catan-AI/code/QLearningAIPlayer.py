import numpy as np
from board import *
from player import *
import pickle

class QLearningAIPlayer(player):
    
    def __init__(self, name, playerColor, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        
        # needs to be moved to some other function once we create the functions to run more, just here for now
        super().__init__(name, playerColor)
        
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.action_key_to_index = {}
        self.index_to_action_key = {}

        #Dev cards in possession
        self.newDevCards = [] #List to keep the new dev cards draw - update the main list every turn
        self.devCards = {'KNIGHT':0, 'VP':0, 'MONOPOLY':0, 'ROADBUILDER':0, 'YEAROFPLENTY':0} 
        self.devCardPlayedThisTurn = False

        self.visibleVictoryPoints = self.victoryPoints - self.devCards['VP']

    # This is currently taken from the heuristic so its random, need to fix
    def updateAI(self): 
        self.isAI = True
        self.setupResources = [] #List to keep track of setup resources
        #Initialize resources with just correct number needed for set up
        self.resources = {'ORE':0, 'BRICK':4, 'WHEAT':2, 'WOOD':4, 'SHEEP':2} #Dictionary that keeps track of resource amounts
        print("Added new AI Player:", self.name)
    
    def get_initial_setup_state(self, board):
        resources_list = []
        coords_list = []

        for index, hex_tile in board.hexTileDict.items():
            # Extract resource and coord attributes from each hexTile object
            resource = hex_tile.resource
            coord = hex_tile.coord
        
            # Append extracted attributes to respective lists
            resources_list.append(resource)
            coords_list.append(coord)
    
        # Convert lists to tuples
        self.resources_tuple = tuple(resources_list)
        self.coords_tuple = tuple(coords_list)

        ports_list = []
        is_colonised_list = []
    
        # Iterate over the dictionary containing hexTile.Vertex objects
        for vertex, hex_vertex in board.boardGraph.items():
            # Extract port and isColonised attributes from each hexTile.Vertex object
            port = hex_vertex.port
            is_colonised = hex_vertex.isColonised
        
            #  Append extracted attributes to respective lists
            ports_list.append(port)
            is_colonised_list.append(is_colonised)
        # Convert lists to tuples
        ports_tuple = tuple(ports_list)
        is_colonised_tuple = tuple(is_colonised_list)

        state = (
        self.settlementsLeft,
        self.roadsLeft,
        self.citiesLeft,
        self.resources_tuple,
        self.coords_tuple,
        tuple(self.resources.values()),
        self.knightsPlayed,
        self.maxRoadLength,
        self.victoryPoints,
        self.longestRoadFlag,
        self.largestArmyFlag,
        ports_tuple, 
        is_colonised_tuple,
        # again need to do something about the board
        )
        return state

    def initial_setup(self, board):
        possible_actions = {}
        state = self.get_initial_setup_state(board)
        possibleVertices = board.get_setup_settlements(self)
        settlement_commands = self.get_settlement_build_commands(possibleVertices)
        possible_actions.update(settlement_commands)

        self.update_action_mappings(list(possible_actions.keys()))

        action_index = self.initial_choose_action(state, board, possible_actions)
        action_key = self.index_to_action_key.get(action_index)
        action_value = possible_actions[action_key]
        print(f"Chosen action: {action_key} with value {action_value}, {action_index}")
        self.build_settlement(action_value, board)

        reward = self.calculate_reward({'type': None})
        next_state = self.get_state(board)
        self.update_q_table(state, action_index, reward, next_state, board)

        possible_actions = {}
        state = self.get_initial_setup_state(board)
        possibleVertices = board.get_setup_roads(self)
        road_commands = self.get_road_build_commands(possibleVertices)
        possible_actions.update(road_commands)

        self.update_action_mappings(list(possible_actions.keys()))

        action_index = self.initial_choose_action(state, board, possible_actions)
        action_key = self.index_to_action_key.get(action_index)
        action_value = possible_actions[action_key]

        print(f"Chosen action: {action_key} with value {action_value}, {action_index}")
        self.build_road(action_value[0], action_value[1], board)

        reward = self.calculate_reward({'type': None})
        next_state = self.get_state(board)
        self.update_q_table(state, action_index, reward, next_state, board)
        
    def initial_choose_action(self, state, board, possible_actions):
        action_keys = list(possible_actions.keys())
        self.update_action_mappings(action_keys)

        # Convert the state to a tuple for use as a key in the Q-table
        state_key = tuple(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(action_keys))
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.choice(len(action_keys))
        else:
            action_index = np.argmax(self.q_table[state_key])
        return action_index
            
    def update_q_table(self, state, action, reward, next_state, board):
        # Convert states to tuples for use as keys in the Q-table
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.get_possible_actions(board)))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.get_possible_actions(board)))

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_delta = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_delta

    def choose_action(self, state, board):
        possible_actions = self.get_possible_actions(board)
        action_keys = list(possible_actions.keys())
        self.update_action_mappings(action_keys)
        
        # Convert the state to a tuple for use as a key in the Q-table
        state_key = tuple(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(action_keys))
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.choice(len(action_keys))
        else:
            action_index = np.argmax(self.q_table[state_key])
        return action_index
        
    def update_action_mappings(self, action_keys):
        self.action_key_to_index = {key: index for index, key in enumerate(action_keys)}
        self.index_to_action_key = {index: key for key, index in self.action_key_to_index.items()}


    def get_state(self, board):
        
        ports_list = []
        is_colonised_list = []
        for vertex, hex_vertex in board.boardGraph.items():
            # Extract port and isColonised attributes from each hexTile.Vertex object
            port = hex_vertex.port
            is_colonised = hex_vertex.isColonised
        
            #  Append extracted attributes to respective lists
            ports_list.append(port)
            is_colonised_list.append(is_colonised)
        # Convert lists to tuples
        ports_tuple = tuple(ports_list)
        is_colonised_tuple = tuple(is_colonised_list)

        state = (
            self.settlementsLeft,
            self.roadsLeft,
            self.citiesLeft,
            self.resources_tuple,
            self.coords_tuple,
            tuple(self.resources.values()),
            self.knightsPlayed,
            self.maxRoadLength,
            self.victoryPoints,
            self.longestRoadFlag,
            self.largestArmyFlag,
            ports_tuple, 
            is_colonised_tuple,
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
        
        # Check if the player has enough resources to trade with bank
        trade_commands = self.trade()
        if trade_commands:
            possible_actions.update(trade_commands)

        # Option to draw a development card if resources are sufficient
        dev_card_draw_commands = {}
        if self.resources['ORE'] >= 1 and self.resources['WHEAT'] >= 1 and self.resources['SHEEP'] >= 1:
            dev_card_draw_commands['draw_dev_card'] = None
            possible_actions.update(dev_card_draw_commands)

        # Option to play a development card if any available and not already played this turn
        if not self.devCardPlayedThisTurn:
            dev_card_play_commands = self.get_play_dev_card_commands()
            possible_actions.update(dev_card_play_commands)
            

        possible_actions['end_turn'] = None

        return possible_actions
    
    #Wrapper function to control all trading
    def trade(self):
        trade_commands = {}
        for r1, r1_amount in self.resources.items():
            if r1_amount >= 6:  
                for r2, r2_amount in self.resources.items():
                    if r2_amount < 1:
                        trade_commands[f'trade_{r1}_for_{r2}'] = (r1, r2)
                        break
        return trade_commands


    

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
    
    def get_robber_commands(self, robber_spots):
        robbing_commands = {}
        command_index = 1
        for hex_ind, hexTile in robber_spots.items():
            vertexList = polygon_corners(self.board.flat, hexTile.hex)
            for vertex in vertexList:
                playerAtVertex = self.board.boardGraph[vertex].state['Player']
                if playerAtVertex and playerAtVertex != self:
                    robbing_commands[f'move_robber_{command_index}'] = (hex_ind, playerAtVertex)
                    command_index += 1
        return robbing_commands

    def calculate_reward(self, game_event):
        #the input is a dictionary {'type': value}

        # Need to figure out a better reward value
        reward = -1
        if game_event['type'] == 'victory_point':
            reward += 100
        elif game_event['type'] == 'settlement':
            reward += 100
        elif game_event['type'] == 'city':
            reward += 100
        elif game_event['type'] == 'discard_card':
            reward -= 10
        elif game_event['type'] == 'trade':
            reward += 10
        return reward


    def move(self, board):
        state = self.get_state(board)
        possible_actions = self.get_possible_actions(board)
        action_index = self.choose_action(state, board)
        action_key = self.index_to_action_key[action_index]
        action_value = possible_actions[action_key]
        print(f"Chosen action: {action_key} with value {action_value}")

        if action_key.startswith("build_settlement"):
            self.build_settlement(action_value, board)
            reward = self.calculate_reward({'type': 'settlement'})
        elif action_key.startswith("build_road"):
            self.build_road(action_value[0], action_value[1], board)
            reward = self.calculate_reward()
        elif action_key.startswith("build_city"):
            self.build_city(action_value, board)
            reward = self.calculate_reward({'type': 'city'})
        elif action_key.startswith("trade_"):
            self.trade_with_bank(action_value[0], action_value[1])
            reward = self.calculate_reward({'type': 'trade'})
        elif action_key.startswith("play_"):
            self.play_devCard(action_value, board)
            reward = self.calculate_reward({'type': 'dev'})
        else:
            reward = 0 

        next_state = self.get_state(board)
        self.update_q_table(state, action_index, reward, next_state, board)

        #Function to trade with bank
    def trade_with_bank(self, r1, r2):
        '''Function to implement trading with bank
        r1: resource player wants to trade away
        r2: resource player wants to receive
        Automatically give player the best available trade ratio
        '''
        #Get r1 port string
        r1_port = "2:1 " + r1
        if(r1_port in self.ports_tuple and self.resources[r1] >= 2): #Can use 2:1 port with r1
            self.resources[r1] -= 2
            self.resources[r2] += 1
            print("Traded 2 {} for 1 {} using {} Port".format(r1, r2, r1))
            return

        #Check for 3:1 Port
        elif('3:1 PORT' in self.ports_tuple and self.resources[r1] >= 3):
            self.resources[r1] -= 3
            self.resources[r2] += 1
            print("Traded 3 {} for 1 {} using 3:1 Port".format(r1, r2))
            return

        #Check 4:1 port
        elif(self.resources[r1] >= 4):
            self.resources[r1] -= 4
            self.resources[r2] += 1
            print("Traded 4 {} for 1 {}".format(r1, r2))
            return
        
        else:
            print("Insufficient resource {} to trade with Bank".format(r1))
        return

    #Function to discard cards

       
    def discardResources(self):
        maxCards = 7 #Default is 7, but can be changed for testing

        #Calculate resources to discard
        totalResourceCount = 0
        for resource, amount in self.resources.items():
            totalResourceCount += amount

        #Logic to calculate number of cards to discard and allow player to select
        if totalResourceCount > maxCards:
            numCardsToDiscard = int(totalResourceCount/2)
            print("\nPlayer {} has {} cards and will have {} cards discarded...".format(self.name, totalResourceCount, numCardsToDiscard))
            
            #Loop to allow player to discard cards
            for i in range(numCardsToDiscard+1):
                print("Player {} current resources to discard from:", self.resources)
                
                resourceToDiscard = max(self.resources, key=self.resources.get)

                #Discard that resource
                self.resources[resourceToDiscard] -= 1


        else:
            print("\nPlayer {} has {} cards and does not need to discard any cards!".format(self.name, totalResourceCount))
            return
    

    def Qlearning_move_robber(self, board):
        state = self.get_state(board)
        robber_spots = board.get_robber_spots()
        robbing_commands = self.get_robber_commands(robber_spots)
        if not robbing_commands:
            return

        self.update_action_mappings(list(robbing_commands.keys()))
        action_index = self.choose_action(state, board)
        action_key = self.index_to_action_key[action_index]
        hexToRob_index, playerToRob = robbing_commands[action_key]

        # Move the robber
        self.move_robber(hexToRob_index, board, playerToRob)

        reward = 2  # Reward for robbing successfully
        next_state = self.get_state(board)
        self.update_q_table(state, action_index, reward, next_state, board)
    
    def choose_player_to_rob(self, board):
        '''Heuristic function to choose the player with maximum points.
        Choose hex with maximum other players, Avoid blocking own resource
        args: game board object
        returns: hex index and player to rob
        '''
        #Get list of robber spots
        #possible_actions = {}
        #state = self.get_state(board)
        robberHexDict = board.get_robber_spots()
        #robbing_commands = self.get_robber_commands(robberHexDict)
        #possible_actions.update(robbing_commands)
        #action_index = self.initial_choose_action(state, board, possible_actions)
        #action_key = self.index_to_action_key[action_index]
        #action_value = possible_actions[action_key]
        #print(f"Chosen action: {action_key} with value {action_value}")
        
        #reward = self.calculate_reward()

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
    
    #Return dictionary of dev cards to play
    def get_play_dev_card_commands(self):
        dev_card_play_commands = {}
        for card_name, card_amount in self.devCards.items():
            if card_name != 'VP' and card_amount > 0:
                if card_name in ['MONOPOLY', 'YEAROFPLENTY']:
                    resources = ['BRICK', 'WOOD', 'WHEAT', 'SHEEP', 'ORE']
                    for resource in resources:
                        if card_name == 'MONOPOLY':
                            dev_card_play_commands[f'play_{card_name.lower()}_{resource.lower()}'] = (card_name, resource)
                        elif card_name == 'YEAROFPLENTY':
                            for second_resource in resources:
                                if second_resource != resource:
                                    dev_card_play_commands[f'play_{card_name.lower()}_{resource.lower()}_{second_resource.lower()}'] = (card_name, (resource, second_resource))
                else:
                    dev_card_play_commands[f'play_{card_name.lower()}'] = card_name
        return dev_card_play_commands
    
    #Function to do the action of playing a dev card
    def play_devCard(self, card_name, game):
        'Update game state'
        # Check if player can play a devCard this turn
        if self.devCardPlayedThisTurn:
            print('Already played 1 Dev Card this turn!')
            return

        if self.devCards[card_name] <= 0:
            print(f'No {card_name} cards available to play!')
            return

        if isinstance(card_name, tuple):
            card_name, resources = card_name
        else:
            card_name = card_name
            resources = None

        self.devCardPlayedThisTurn = True
        self.devCards[card_name] -= 1

        print("Playing Dev Card:", card_name)

        # Logic for each Dev Card
        if card_name == 'KNIGHT':
            game.robber(self)
            self.knightsPlayed += 1 

        elif card_name == 'ROADBUILDER':
            game.build(self, 'ROAD')
            game.boardView.displayGameScreen()
            game.build(self, 'ROAD')
            game.boardView.displayGameScreen()

        elif card_name == 'YEAROFPLENTY':
            resource1, resource2 = resources
            self.resources[resource1] += 1
            self.resources[resource2] += 1
            print(f"Received 1 {resource1} and 1 {resource2} from YEAROFPLENTY")

        elif card_name == 'MONOPOLY':
            resource_to_monopolize = resources
            for player in list(game.playerQueue.queue):
                if player != self:
                    num_lost = player.resources[resource_to_monopolize]
                    player.resources[resource_to_monopolize] = 0
                    self.resources[resource_to_monopolize] += num_lost
            print(f"Monopolized all {resource_to_monopolize} resources from opponents")
            #Need to implement how to decide which resoures to select
            #Player picks a resource and gets all opponents resource cards of that type

            
        return


    # Here's a potential solution to the saving q-values procedure, save and reload them every time
    def save_q_values(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_values(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No existing Q-values file found. Starting with a new Q-table.")

    def train(self, num_episodes):
        for episode in range(num_episodes):
            # Training logic
            ...
            self.save_q_values('q_values.pkl')

    def play_game(self):
        # Load Q-values before starting a new game
        self.load_q_values('q_values.pkl')