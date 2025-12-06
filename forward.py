import numpy as np
from typing import List, Tuple, TypeVar, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm

A = TypeVar('A')  # Action type
KnownState = TypeVar('KnownState')  # Known state type (generic)

@dataclass(frozen=True)
class State:
    """
    Complete state representation with uncertain and known components.
    """
    uncertain: int  # Index into the discrete state space
    known: KnownState  # Known state variables


@dataclass
class BeliefState:
    """
    Hybrid belief state that combines:
    1. Discrete probability distribution over uncertain state components
    2. Known (deterministic) state components
    """
    belief: np.ndarray  # Probability distribution over uncertain states
    known_state: KnownState  # Known state variables
    
    def __post_init__(self):
        """Validate and ensure belief is a numpy array."""
        self.belief = np.asarray(self.belief, dtype=float)
        if not np.isclose(self.belief.sum(), 1.0):
            raise ValueError(f"Belief probabilities must sum to 1, got {self.belief.sum()}")
        if np.any(self.belief < 0):
            raise ValueError("Belief probabilities must be non-negative")
    
    def update(self, 
               observation: int,
               likelihood_obs_given_state: np.ndarray,
               new_known_state: KnownState) -> 'BeliefState':
        """
        Perform Bayesian update and return new belief state.
        
        Args:
            observation: Observation index (int)
            likelihood_obs_given_state: Array where likelihood_obs_given_state[i] = P(observation | uncertain_state=i)
            new_known_state: Updated known state
        
        Returns:
            New BeliefState with updated belief and known state
        """
        # Bayesian update: posterior ∝ prior × likelihood
        posterior = self.belief * likelihood_obs_given_state
        
        # Normalize
        if posterior.sum() > 0:
            posterior = posterior / posterior.sum()
        else:
            # If all likelihoods are zero, keep prior
            posterior = self.belief.copy()
        
        return BeliefState(
            belief=posterior,
            known_state=new_known_state
        )
    
    def expected_reward(self, action: A, reward_fn) -> float:
        """
        Calculate expected reward for taking an action from this belief state.
        
        Args:
            action: Action
            reward_fn: Function that takes (State, action) and returns reward (float)
        
        Returns:
            Expected reward: E[R(s,a)] = Σ_s b(s) * R(s,a)
        """
        expected_r = 0.0
        
        for state_idx in range(len(self.belief)):
            state = State(uncertain=state_idx, known=self.known_state)
            reward = reward_fn(state, action)
            expected_r += self.belief[state_idx] * reward
        
        return expected_r


class Game(ABC):
    """
    Abstract base class defining the game dynamics for forward search.
    """
    
    @abstractmethod
    def actions(self, belief_state: BeliefState) -> List[A]:
        """
        Get list of valid actions from a belief state.
        
        Args:
            belief_state: Current belief state (contains both belief and known_state)
            
        Returns:
            List of valid actions
        """
        pass
    
    @abstractmethod
    def reward(self, state: State, action: A) -> float:
        """
        Get reward for taking action from a state.
        
        Args:
            state: Complete state (uncertain + known)
            action: Action taken
            
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def observation_probs(self, action: A) -> np.ndarray:
        """
        Get observation probability matrix for all uncertain states given an action.
        
        Args:
            action: Action taken
            
        Returns:
            Matrix of shape (n_observations, n_uncertain_states) where entry [i, j] is
            P(observation=i | uncertain_state=j, action)
            
            This is a square matrix for n_uncertain_states == n_observations.
        """
        pass
    
    @abstractmethod
    def transition(self, known_state: KnownState, action: A) -> KnownState:
        """
        Get new known state after taking action. Uncertain state remains fixed.
        
        Args:
            known_state: Current known state
            action: Action taken
            
        Returns:
            New known state
        """
        pass

    def step(self, state: State, action: A) -> Tuple[State, int, float]:
        """
        Execute action in the environment: transition, sample observation, get reward.
        
        This is a convenience method that combines the core dynamics into a single step,
        useful for simulation and rollouts.
        
        Args:
            state: Current complete state (uncertain + known)
            action: Action to take
            
        Returns:
            Tuple of (new_state, observation, reward) where:
            - new_state: New complete state (uncertain state unchanged, known state updated)
            - observation: Sampled observation index
            - reward: Reward received for taking action from state
        """
        # Get reward for current state-action pair
        reward = self.reward(state, action)
        
        # Transition the known state
        new_known_state = self.transition(state.known, action)
        
        # Sample observation from P(o | uncertain_state, action)
        obs_probs_matrix = self.observation_probs(action)
        obs_probs_given_state = obs_probs_matrix[:, state.uncertain]
        observation = np.random.choice(len(obs_probs_given_state), p=obs_probs_given_state)
        
        # Create new state (uncertain state remains the same)
        new_state = State(uncertain=state.uncertain, known=new_known_state)
        
        return new_state, observation, reward


class ForwardSearch:
    """
    Forward search for POMDPs using the algorithm from equation 22.1.
    
    Searches the action-observation-belief tree to a specified depth
    to select the action with highest expected value.
    
    For our problem, observations are indices into the belief state
    (n_observations = len(belief)).
    """
    
    def __init__(self, 
                 game: Game, 
                 discount: float = 1.0,
                 value_fn: Optional[callable] = None):
        """
        Initialize forward search.
        
        Args:
            game: Game instance defining dynamics
            discount: Discount factor γ (gamma), default 1.0
            value_fn: Optional value function for leaf nodes (depth 0)
                     Takes BeliefState and returns float
                     Default: maximum expected immediate reward
        """
        self.game = game
        self.discount = discount
        self.value_fn = value_fn if value_fn is not None else self._default_value_fn
    
    def _default_value_fn(self, belief_state: BeliefState) -> float:
        """
        Default value function: max expected immediate reward.
        """
        valid_actions = self.game.actions(belief_state)
        if not valid_actions:
            return 0.0
        
        max_value = float('-inf')
        for action in valid_actions:
            value = belief_state.expected_reward(action, self.game.reward)
            if value > max_value:
                max_value = value
        return max_value
    
    def Q_value(self, belief_state: BeliefState, action: A, depth: int) -> float:
        """
        Compute Q_d(b, a) - the value of taking action a from belief b at depth d.
        
        This implements equation 22.1:
        Q_d(b,a) = R(b,a) + gamma Σ_o P(o|b,a) U_{d-1}(Update(b,a,o))  if d > 0
                 = U(b)                                              otherwise
        
        Args:
            belief_state: Current belief state
            action: Action to evaluate
            depth: Remaining search depth
        
        Returns:
            Q-value for (belief_state, action) at given depth
        """
        if depth == 0:
            # Base case: use approximate value function
            return self.value_fn(belief_state)
        
        # Recursive case: compute immediate reward + expected future value
        immediate_reward = belief_state.expected_reward(action, self.game.reward)
        
        # Number of observations = size of belief (observe which discrete uncertain state)
        n_observations = len(belief_state.belief)
        
        # Get new known state after taking action (independent of observation)
        new_known_state = self.game.transition(belief_state.known_state, action)
        
        # Get observation probability matrix: obs_prob_matrix[o, s] = P(o | s, a)
        # Shape: (n_observations, n_uncertain_states)
        obs_prob_matrix = self.game.observation_probs(action)
        
        # Compute P(o | b, a) = Σ_s b(s) * P(o | s, a) for all observations at once
        # Matrix-vector multiply: obs_prob_matrix @ belief gives P(o|b,a) for each observation
        prob_obs_given_belief = obs_prob_matrix @ belief_state.belief  # Shape: (n_observations,)
        
        expected_future_value = 0.0
        for observation in range(n_observations):
            if prob_obs_given_belief[observation] > 0:  # Only consider observations with non-zero probability
                # Get likelihood for Bayesian update: P(o | s, a) for all states s
                # This is the observation-th row of the observation probability matrix
                likelihood_obs_given_state = obs_prob_matrix[observation, :]
                
                # Create updated belief state
                updated_belief = belief_state.update(observation, likelihood_obs_given_state, new_known_state)
                
                # Recursively compute value of updated belief
                future_value = self.U_value(updated_belief, depth - 1)
                
                expected_future_value += prob_obs_given_belief[observation] * future_value
        
        return immediate_reward + self.discount * expected_future_value
    
    def U_value(self, belief_state: BeliefState, depth: int) -> float:
        """
        Compute U_d(b) = max_a Q_d(b, a).
        
        Args:
            belief_state: Current belief state
            depth: Remaining search depth
        
        Returns:
            Value of belief state at given depth
        """
        if depth == 0:
            return self.value_fn(belief_state)
        
        valid_actions = self.game.actions(belief_state)
        if not valid_actions:
            return 0.0
        
        max_q_value = float('-inf')
        for action in valid_actions:
            q_value = self.Q_value(belief_state, action, depth)
            if q_value > max_q_value:
                max_q_value = q_value
        
        return max_q_value

    def search(self, belief_state: BeliefState, depth: int) -> Tuple[A, float]:
        """
        Perform forward search and return the best action.
        
        Args:
            belief_state: Current belief state
            depth: Search depth
        
        Returns:
            Tuple of (best_action, best_q_value)
        """
        valid_actions = self.game.actions(belief_state)
        if not valid_actions:
            return None, 0.0
        
        best_action = None
        best_q_value = float('-inf')
        
        for action in tqdm(valid_actions, desc=f"Search depth {depth}", leave=False):
            q_value = self.Q_value(belief_state, action, depth)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action, best_q_value
    
    def get_action_values(self, belief_state: BeliefState, depth: int) -> List[Tuple[A, float]]:
        """
        Get Q-values for all valid actions from current belief state.
        
        Args:
            belief_state: Current belief state
            depth: Search depth
        
        Returns:
            List of (action, q_value) tuples
        """
        valid_actions = self.game.actions(belief_state)
        return [(action, self.Q_value(belief_state, action, depth)) for action in valid_actions]