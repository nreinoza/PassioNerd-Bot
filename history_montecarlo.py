import random
from game import Game

ROLLOUT_DEPTH = 3

#Referenced ch07.py from griffinbholt's GitHub of Python translations of textbook code snippets
class POMDP():
    """
    gamma
    S #state space 
    A #action space
    O #don't need for MCTS
    T #don't need for MCTS 
    R #don't need for MCTS
    O #don't need for MCTS
    sample_tro #A generative model for a state, reward, and observation, given a state and action
    """

    def __init__(self, gamma, S, A, sample_tro):
        self.gamma = gamma
        self.S = S 
        self.A = A 
        self.sample_tro = sample_tro 

    def randstep(self, h, a) -> tuple[Any, float]:
        return self.sample_tro(s, a) #need to confirm if this is accurate
        

#Referenced ch09.py from griffinbholt's GitHub of Python translations of textbook code snippets & Algorithm 22.1 in Chapter 22 of textbook
class MonteCarloTreeSearch(OnlinePlanningMethod):
    def __init__(self, P:POMDP, N, Q, depth, num_sims, exploration_const, val_approx_func):
        self.P = P #problem package
        self.N = N #visit counts
        self.Q = Q #action-value estimates
        self.d = depth
        self.m = num_sims
        self.c = exploration_const
        self.U = val_approx_func #Idea: use rollout for value function approximation?
    
    #Referenced ch19.py from griffinbholt's GitHub of Python translations of textbook code snippets
    def __call__(self, b, h) -> Any: #pass in a uniform unitial belief distribution, and a blank history to be used for each simulation 
        A = self.P.A
        for i in self.m:
            s = random.choices(self.P.S, weghts = b) #Use some kind of random state sampling to obtain starting state s
            self.simulate(s, h, d = self.d)
        return A[np.argmax([self.Q[(h, a)] for a in A])]

    def simulate(self, s:Any, h:Any, d:int): 
        if d <= 0:
            return self.U(s)
        if (h, self.P.A[0]) not in self.N: #found an unexplored history-action pair
            for a in self.P.A: 
                self.N[(h, a)] = 0 
                self.Q[(h, a)] = 0.0
            return self.U(s) #estimate value of unexplored node

        a = self.explore(h) #use exploration strategy to choose an action
        s_prime, r, o = self.P.sample_tro(s, a) 
        q = r + self.P.gamma * self.simulate(s_prime, np.vstack(h, (a, o)), d - 1) #recursively calculate discounted reward
        self.N[(h, a)] += 1 #increment visit count
        self.Q[(h, a)] += (q - self.Q[(h, a)]) / self.N[(h, a)] #update action-value estimate
        return q

    def explore(self, h:Any) -> Any:
        A, N = self.P.A, self.N
        Nh = np.sum([N(h, a) for a in A]) 
        return A[np.argmax([self.ucb1(h, a, Nh) for a in A])] #pick action with highest ucb1 score

    def ucb1(self, h:Any, a:Any, Nh:int) -> float:
        N, Q, c = self.N, self.Q, self.c
        return Q[(h, a)] + c * self.bonus(N[(h, a)], Nh)

    @staticmethod
    def bonus(Nha: int, Nh: int) -> float:
        return np.inf if Nha == 0 else np.sqrt(np.log(Nh)/Nha)


#Referenced ch09.py from griffinbholt's GitHub of Python translations of textbook code snippets
def rollout(P: POMDP, state, rollout_policy, ROLLOUT_DEPTH) -> float:
    ret = 0.0
    for d in range ROLLOUT_DEPTH:
        action = rollout_policy(state)
        state, reward = P.randstep(state, action)
        ret += reward
    return ret