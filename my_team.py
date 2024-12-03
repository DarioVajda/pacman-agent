# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.returning = False

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # Get the current food carrying count
        current_food_carrying = game_state.get_agent_state(self.index).num_carrying

        # Get whether the agent is on its own side
        is_on_own_side = not game_state.get_agent_state(self.index).is_pacman

        # If carrying 2 or more food, enter "return mode"
        if current_food_carrying >= 2:
            self.returning = True

        # If in "return mode", move to cross the dividing line
        if self.returning:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)

                # Minimize distance to any position on the agent's side
                if self.is_on_own_side(pos2):
                    dist = 0  # Already on the correct side
                else:
                    dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist

            # If the agent is on its side, stop returning and resume offensive behavior
            if is_on_own_side:
                self.returning = False
            return best_action

        # Normal behavior: Evaluate actions based on features and weights
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def is_on_own_side(self, position):
        """
        Determines if a given position is on the agent's own side of the board.
        """
        x, y = position
        if self.red:
            return x < self.start[0]  # Red side is the left half of the map
        else:
            return x >= self.start[0]  # Blue side is the right half of the map
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free with enhanced patrol behavior.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.patrol_positions = []  # Define patrol positions dynamically
        self.patrol_index = 0  # To cycle through patrol points dynamically

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.define_patrol_area(game_state)

    def define_patrol_area(self, game_state):
        """
        Defines patrol points in the 4 columns closest to the middle on the agent's side.
        """
        mid_x = game_state.data.layout.width // 2
        walls = game_state.get_walls()

        # Determine patrol columns based on the agent's team
        if self.red:
            patrol_columns = range(mid_x - 4, mid_x)  # 4 columns on the left side
        else:
            patrol_columns = range(mid_x, mid_x + 4)  # 4 columns on the right side

        # Identify all valid positions in these columns
        self.patrol_positions = [
            (x, y) for x in patrol_columns for y in range(walls.height) if not walls[x][y]
        ]
        random.shuffle(self.patrol_positions)  # Shuffle patrol points for varied movement

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Determine whether the agent is on defense
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Compute distance to visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Chase the nearest visible invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # Patrol the middle columns on the agent's side
            patrol_target = self.patrol_positions[self.patrol_index]
            features['distance_to_patrol'] = self.get_maze_distance(my_pos, patrol_target)

        # Penalize stopping or reversing direction
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Weights for the features to guide behavior.
        """
        return {
            'num_invaders': -1000,     # Strongly prioritize chasing invaders
            'on_defense': 100,        # Stay on defense
            'invader_distance': -10,  # Get close to invaders
            'distance_to_patrol': -5, # Patrol the middle area on its side
            'stop': -100,             # Avoid stopping
            'reverse': -2             # Discourage reversing direction
        }

    def choose_action(self, game_state):
        """
        Chooses an action based on feature evaluation.
        """
        actions = game_state.get_legal_actions(self.index)

        # Cycle through patrol points if no invaders are visible
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if len(invaders) == 0:
            # Advance to the next patrol position
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_positions)

        # Evaluate actions and select the best one
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)