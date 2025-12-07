import numpy as np
from typing import List, Tuple, TypeVar, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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


def _compute_q_value_worker(action: A, forward_search: 'ForwardSearch', 
                            belief_state: BeliefState, depth: int) -> Tuple[A, float]:
    """
    Worker function to compute Q-value for a single action.
    This is a module-level function to support multiprocessing pickling.
    
    Args:
        action: Action to evaluate
        forward_search: ForwardSearch instance
        belief_state: Current belief state
        depth: Search depth
    
    Returns:
        Tuple of (action, q_value)
    """
    q_value = forward_search.Q_value(belief_state, action, depth)
    return action, q_value


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
                 value_fn: Optional[callable] = None,
                 n_processes: Optional[int] = None):
        """
        Initialize forward search.
        
        Args:
            game: Game instance defining dynamics
            discount: Discount factor γ (gamma), default 1.0
            value_fn: Optional value function for leaf nodes (depth 0)
                     Takes BeliefState and returns float
                     Default: maximum expected immediate reward
            n_processes: Number of parallel processes to use (default: cpu_count())
        """
        self.game = game
        self.discount = discount
        self.value_fn = value_fn if value_fn is not None else self._default_value_fn
        self.n_processes = n_processes if n_processes is not None else cpu_count()
    
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

    @staticmethod
    def units_bonus_value_fn(game: Game, max_bonus: float = 100.0) -> callable:
        """
        Create a value function that combines immediate reward with units completion bonus
        and progression bonuses for intermediate and advanced courses.
        
        This encourages the agent to make progress by rewarding:
        - Number of units completed (normalized to target)
        - Intermediate courses taken (up to 3)
        - Advanced courses taken (up to 2)
        
        The units bonus is normalized so that:
        - 0 units = 0 bonus
        - Target units (156) = max_bonus
        - Bonus caps at max_bonus once target is reached
        
        Args:
            game: Game instance (needed to access course metadata)
            max_bonus: Maximum bonus at target units (default: 60.0)
        
        Returns:
            Value function that takes BeliefState and returns float
        
        Example usage:
            value_fn = ForwardSearch.units_bonus_value_fn(game, max_bonus=60.0)
            forward_search = ForwardSearch(game, value_fn=value_fn)
        """
        # Calculate target units: MAX_QUARTERS * 15 - (4 * MAX_QUARTERS / 2)
        # = 12 * 15 - (4 * 6) = 180 - 24 = 156 units
        max_quarters = game.MAX_QUARTERS
        target_units = max_quarters * 15 - (4 * max_quarters / 2)
        
        def value_fn(belief_state: BeliefState) -> float:
            # Get best immediate expected reward
            valid_actions = game.actions(belief_state)
            if not valid_actions:
                return 0.0
            
            max_immediate_reward = float('-inf')
            for action in valid_actions:
                reward = belief_state.expected_reward(action, game.reward)
                if reward > max_immediate_reward:
                    max_immediate_reward = reward
            
            # Calculate units completion bonus
            taken_course_ids = set()
            for quarter in belief_state.known_state:
                taken_course_ids.update(quarter)
            
            total_units = sum(
                game.course_metadata[cid]['units'] 
                for cid in taken_course_ids
            )
            
            # Normalize units bonus: linear scaling up to target, then cap
            if total_units >= target_units:
                units_bonus = max_bonus
            else:
                units_bonus = (total_units / target_units) * max_bonus
            
            # Count intermediate and advanced courses
            counts = game._count_courses_by_subject(belief_state.known_state)
            
            num_intermediate = 0
            num_advanced = 0
            for row in counts.iter_rows(named=True):
                if row['level'] == 1:
                    num_intermediate += row['count']
                elif row['level'] == 2:
                    num_advanced += row['count']
            
            # Progression bonuses (cap at max)
            intermediate_bonus = min(num_intermediate, 3) * 20  # 20 points per intermediate, max 3
            advanced_bonus = min(num_advanced, 2) * 50  # 50 points per advanced, max 2
            
            return max_immediate_reward + units_bonus + intermediate_bonus + advanced_bonus
        
        return value_fn
    
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
        
        # Filter to only consider observations with sufficient probability
        threshold = 0.032
        filtered_probs = prob_obs_given_belief.copy()
        filtered_probs[filtered_probs <= threshold] = 0
        
        # Renormalize after filtering
        total_prob = filtered_probs.sum()
        if total_prob > 0:
            filtered_probs = filtered_probs / total_prob
        else:
            # If all filtered out, keep only the most likely observation
            filtered_probs = np.zeros_like(prob_obs_given_belief)
            filtered_probs[prob_obs_given_belief.argmax()] = 1.0
        
        expected_future_value = 0.0
        for observation in range(n_observations):
            if filtered_probs[observation] == 0:
                continue
                
            # Get likelihood for Bayesian update: P(o | s, a) for all states s
            # This is the observation-th row of the observation probability matrix
            likelihood_obs_given_state = obs_prob_matrix[observation, :]
                
            # Create updated belief state
            updated_belief = belief_state.update(observation, likelihood_obs_given_state, new_known_state)
                
            # Recursively compute value of updated belief
            future_value = self.U_value(updated_belief, depth - 1)
                
            # Use the RENORMALIZED probability
            expected_future_value += filtered_probs[observation] * future_value
        
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

    def search(self, belief_state: BeliefState, depth: int) -> Tuple[A, float, List[Tuple[A, float]]]:
        """
        Perform forward search and return the best action along with all Q-values.
        Uses multiprocessing for parallel Q-value computation when depth > 1.
        
        Args:
            belief_state: Current belief state
            depth: Search depth
        
        Returns:
            Tuple of (best_action, best_q_value, all_action_values) where
            all_action_values is a list of (action, q_value) tuples sorted by q_value descending
        """
        valid_actions = self.game.actions(belief_state)
        if not valid_actions:
            return None, 0.0, []
        
        # Use multiprocessing if depth > 1 and we have multiple actions
        if depth > 2 and len(valid_actions) > 1:
            # Create a partial function with fixed parameters
            worker_fn = partial(_compute_q_value_worker, 
                            forward_search=self,
                            belief_state=belief_state, 
                            depth=depth)
            
            # Use multiprocessing pool to compute Q-values in parallel
            with Pool(processes=self.n_processes) as pool:
                results = list(tqdm(
                    pool.imap(worker_fn, valid_actions),
                    total=len(valid_actions),
                    desc=f"Search depth {depth}",
                    leave=False
                ))
        else:
            # Sequential execution for shallow searches
            results = []
            for action in tqdm(valid_actions, desc=f"Search depth {depth}", leave=False):
                q_value = self.Q_value(belief_state, action, depth)
                results.append((action, q_value))
        
        # Sort by Q-value descending
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Best action is the first one
        best_action, best_q_value = results_sorted[0]
        
        return best_action, best_q_value, results_sorted