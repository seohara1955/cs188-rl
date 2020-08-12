# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = util.Counter() # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return float(self.qvalues[state,action])

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        best_value = 0.0
        legalActions = self.getLegalActions(state)
        if legalActions:
            best_value =\
                max([self.getQValue(state,action) for action in legalActions])
        return best_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  If there are no legal
          actions, which is the case at a terminal state, return None.
        """
        eps = 0.001
        best_action = None
        legalActions = self.getLegalActions(state)
        if legalActions:
            best_value = self.getValue(state)
            best_actions =\
                [action \
                    for action in legalActions \
                        if abs(self.getQValue(state,action) - best_value) < eps]
            best_action = random.choice(best_actions)
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, take a random action otherwise,
          take the best policy action.  If there are no legal actions, 
          which is the case at a terminal state, choose None as the action.
        """
        action = None
        if util.flipCoin(self.epsilon):
              action = random.choice(self.getLegalActions(state))
        else:
              action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state => action => nextState and reward transition.
          NEVER call this function. It is called on your behalf.
        """
        self.qvalues[state,action] = \
            (1-self.alpha)*self.getQValue(state,action)\
                + self.alpha*(reward + self.discount*self.getValue(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class DynaQAgent(QLearningAgent):
    """ 
        Implement the DynaQ reinforcement learning algorithm
        Note: code adapted from Steven Aronson
    """

    def __init__(self, planning_steps = 10, **args):
        QLearningAgent.__init__(self, **args)
        self.planning_steps = planning_steps
        self.model = {}
        
    def setModel(self, state, action, next_state, reward):
        self.model[(state, action)] = (next_state, reward)

    def getModel(self, state, action):
        return self.model[(state, action)]
    
    def setPlanningSteps(self, steps):
        self.planning_steps = steps

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state => action => nextState and reward transition.
          NEVER call this function. It is called on your behalf.
        """
        self.qvalues[state,action] = \
            (1-self.alpha)*self.getQValue(state,action)\
                + self.alpha*(reward + self.discount*self.getValue(nextState))   
        self.setModel(state, action, nextState, reward)

        # planning phase
        if self.planning_steps > 0:
            for step in range(self.planning_steps):
                (m_state, m_action) = random.choice(list(self.model))
                (next_m_state, m_reward) = self.getModel(m_state, m_action)
                self.qvalues[m_state,m_action] = \
                    (1-self.alpha)*self.getQValue(m_state , m_action)\
                    + self.alpha*(m_reward + self.discount*self.getValue(next_m_state))

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       Only getQValue and update are overridden. All other
       QLearningAgent functions should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Return Q(state,action) = w * featureVector where * is the dotProduct operator
        """
        return float(self.getWeights() * self.featExtractor.getFeatures(state,action))

    def update(self, state, action, nextState, reward):
        """
           Update weights based on transition
        """
        difference = (reward + self.discount*self.getValue(nextState))\
                        - self.getQValue(state,action)
        features = self.featExtractor.getFeatures(state,action)
        for feature in features.keys():
            self.weights[feature] =\
                self.weights[feature]\
                + (self.alpha*difference)*features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print("Print weights here for debug.")
