from __future__ import annotations
import math
from copy import deepcopy
from re import S
import time
from unittest import result
from action import action
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None):
        self.children = {}
        self.parent = parent
        self.results = []
        self.param = param
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        self.select(state);
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        available_actions = state.get_actions();
        best_action = 0;
        for action in available_actions:
            if action in self.children:
                if self.children[action].results > best_action:
                    best_action = self.children[action].results;
            else:
                self.expand(state, available_actions);
        return best_action;
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        print(indent * " " + "Parameters: " + self.param + "\nChildren: " + self.children)
        for child in self.children:
            self.children[child].print_tree(self.children[child], indent + 1);
        pass


    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        available_actions = state.get_actions();
        unexplored_actions = [a for a in available_actions if a not in self.children];
        if unexplored_actions:
            self.expand(state, available_actions);
            return
        total_visits = sum(len(child.results) for child in self.children.values());
        log_total = math.log(total_visits) if total_visits > 0 else 0;

        # UCB-1 implementation below
        ucb_values = {};
        for action, child in self.children.items():
            n = len(child.results);
            if n == 0:
                ucb_values[action] = float('inf');
            else:
                average = sum(child.results) / n;
                total_visits = sum(len(c.results) for c in self.children.values());
                log_total = math.log(total_visits) if total_visits > 0 else 0;
                ucb_values[action] = average + self.param * math.sqrt(log_total/n);
        
        best_action = max(ucb_values, key=ucb_values.get);
        best_child = self.children[best_action];

        next_state = state.copy();
        next_state.appply_action(best_action);
        best_child.select(next_state);
        pass

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available):
        unexplored_actions = [a for a in available if a not in self.children];
        action = random.choice(unexplored_actions);
        self.children[action] = TreeNode(self.param, self);
        self.children[action].rollout(state.apply_action(action));
        pass 

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        while not state.is_terminal():
            actions = state.get_actions();
            action = random.choice(actions);
            state.apply_action(action);
        result = self.score(state);
        self.backpropagate(self, result);
        pass
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.results.append(result);
        if self.parent != None:
            self.parent.backpropagate(result);
        pass
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state): 
        return state.score();
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
