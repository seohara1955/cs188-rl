# sarsaAgents.py
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

from learningAgents import ReinforcementAgent
import random,util

class sarsaAgent(ReinforcementAgent):
    """
      SARSA Agent
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = util.Counter() # A Counter is a dict with default 0
        self.nextAction = None

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
        elif not self.nextAction:
            action = self.getPolicy(state)
        else:
            action = self.nextAction

        return action

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state => action => next_state and reward transition.
          NEVER call this function. It is called on your behalf.
        """
        next_action = self.getPolicy(next_state)
        self.qvalues[state,action] =\
            self.getQValue(state,action)\
                + self.alpha*(reward\
                    + self.discount*self.getQValue(next_state, next_action)\
                    - self.getQValue(state,action))
        self.nextAction = next_action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)