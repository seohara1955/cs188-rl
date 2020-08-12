# ExpSarsaAgents.py
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

class ExpSarsaAgents(ReinforcementAgent):
    """
      Expected SARSA Agent
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

    def computeActionsFromQValues(self, state):
        """
          Compute the best actions to take in a state.  When there is at least
          one legal action, returns best_actions, other_actions where best_actions
          is a list of the best actions and other_actions is a list of the other
          actions that are not the best. If there are no legal actions,
          which is the case at a terminal state, return pair None, None.
        """
        eps = 0.001
        best_actions = None
        other_actions = None
        legalActions = self.getLegalActions(state)
        if legalActions:
            best_value = self.getValue(state)
            best_actions =\
                [action \
                    for action in legalActions \
                        if abs(self.getQValue(state,action) - best_value) < eps]
            other_actions = list(set(legalActions) - set(best_actions))
        return best_actions, other_actions

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

    def getExpectedValue(self, state):
        """
          Returns expected action value Q(state,action) over all
          legal actions weighted by the probability that the action
          will be taken. Note that if there are no legal actions,
          which is the case at the terminal state, you should
          return a value of 0.0.
        """
        # find set of best actions and set of all other actions
        best_actions, other_actions = self.computeActionsFromQValues(state)
        if not best_actions:
            return 0.0

         # each non-best action occurs with probability epsilon
        expected_value = 0.0
        num_actions = len(best_actions) + len(other_actions)
        random_action_probability = self.epsilon/num_actions
        for action in other_actions:
            expected_value += random_action_probability*self.getQValue(state,action)
        
        # each best action occurs with probability (1-epsilon) / num_best_actions + epsilon
        # (1-epsilon) / num_best_actions = probability divided equally among all possible actions
        # epsilon = probability a best action will be randomly chosen.
        num_best_actions = len(best_actions)
        for action in best_actions:
            expected_value +=\
                ((1.0-self.epsilon)/num_best_actions + random_action_probability)\
                    *self.getQValue(state,action)

        return expected_value

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state => action => nextState and reward transition.
          NEVER call this function. It is called on your behalf.
        """
        self.qvalues[state,action] =\
            self.getQValue(state,action)\
            + self.alpha*(reward + self.discount*self.getExpectedValue(nextState) - self.getQValue(state,action))      

    def getPolicy(self, state):
        (actions,_) = self.computeActionsFromQValues(state)
        return random.choice(actions) if actions else None
    
    def getValue(self, state):
        return self.computeValueFromQValues(state)