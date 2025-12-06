import numpy as np
import random
from forward import A, Game, BeliefState
from abc import ABC, abstractmethod

class RandomBaseline:
    """
    A random policy to be used as a baseline comparison for our Forward Search implementation.
    """

    def __init__(self, game: Game, uniform_belief: BeliefState):
        self.game = game
        self.uniform_belief = uniform_belief
    
    def run(self, belief_state: BeliefState) -> A:
        """
        Choose a random course from valid courses
        """
        valid_actions = self.game.actions(belief_state)
        return random.choice(valid_actions) #randomly sample a possible quarter (collection of classes)



