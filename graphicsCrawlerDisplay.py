# graphicsCrawlerDisplay.py
# -------------------------
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


# graphicsCrawlerDisplay.py
# -------------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import tkinter
import qlearningAgents
import time
import threading
import sys
import crawler
import math
from math import pi as PI

class Application:

    def sigmoid(self, x):
        return 1.0 / (1.0 + 2.0 ** (-x))

    def incrementSpeed(self, inc):
        self.tickTime *= inc
        self.speed_label['text'] = 'Step Delay: %.5f' % (self.tickTime)

    def incrementEpsilon(self, inc):
        self.ep += inc
        self.epsilon = self.sigmoid(self.ep)
        self.learner.setEpsilon(self.epsilon)
        self.epsilon_label['text'] = 'Epsilon: %.3f' % (self.epsilon)

    def incrementGamma(self, inc):
        self.ga += inc
        self.gamma = self.sigmoid(self.ga)
        self.learner.setDiscount(self.gamma)
        self.gamma_label['text'] = 'Discount: %.3f' % (self.gamma)

    def incrementAlpha(self, inc):
        self.al += inc
        self.alpha = self.sigmoid(self.al)
        self.learner.setLearningRate(self.alpha)
        self.alpha_label['text'] = 'Learning Rate: %.3f' % (self.alpha)

    def __initGUI(self, win):
        ## Window ##
        self.win = win

        ## Initialize Frame ##
        win.grid()
        self.dec = -.5
        self.inc = .5
        self.tickTime = 0.1

        ## Speed Button + Label ##
        self.setupSpeedButtonAndLabel(win)

        ## Epsilon Button + Label ##
        self.setupEpsilonButtonAndLabel(win)

        ## Gamma Button + Label ##
        self.setUpGammaButtonAndLabel(win)

        ## Alpha Button + Label ##
        self.setupAlphaButtonAndLabel(win)

        ## Exit Button ##
        self.exit_button = tkinter.Button(win,text='Quit', command=self.exit)
        self.exit_button.grid(row=0, column=9)

         ## Canvas ##
        self.canvas = tkinter.Canvas(root, height=200, width=1000)
        self.canvas.grid(row=2,columnspan=10)

    def setupAlphaButtonAndLabel(self, win):
        self.alpha_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementAlpha(self.dec)))
        self.alpha_minus.grid(row=1, column=3, padx=10)

        self.alpha = self.sigmoid(self.al)
        self.alpha_label = tkinter.Label(win, text='Learning Rate: %.3f' % (self.alpha))
        self.alpha_label.grid(row=1, column=4)

        self.alpha_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementAlpha(self.inc)))
        self.alpha_plus.grid(row=1, column=5, padx=10)

    def setUpGammaButtonAndLabel(self, win):
        self.gamma_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementGamma(self.dec)))
        self.gamma_minus.grid(row=1, column=0, padx=10)

        self.gamma = self.sigmoid(self.ga)
        self.gamma_label = tkinter.Label(win, text='Discount: %.3f' % (self.gamma))
        self.gamma_label.grid(row=1, column=1)

        self.gamma_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementGamma(self.inc)))
        self.gamma_plus.grid(row=1, column=2, padx=10)

    def setupEpsilonButtonAndLabel(self, win):
        self.epsilon_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementEpsilon(self.dec)))
        self.epsilon_minus.grid(row=0, column=3)

        self.epsilon = self.sigmoid(self.ep)
        self.epsilon_label = tkinter.Label(win, text='Epsilon: %.3f' % (self.epsilon))
        self.epsilon_label.grid(row=0, column=4)

        self.epsilon_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementEpsilon(self.inc)))
        self.epsilon_plus.grid(row=0, column=5)

    def setupSpeedButtonAndLabel(self, win):
        self.speed_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementSpeed(.5)))
        self.speed_minus.grid(row=0, column=0)

        self.speed_label = tkinter.Label(win, text='Step Delay: %.5f' % (self.tickTime))
        self.speed_label.grid(row=0, column=1)

        self.speed_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementSpeed(2)))
        self.speed_plus.grid(row=0, column=2)

    def skip5kSteps(self):
        self.stepsToSkip = 5000

    def __init__(self, win):

        self.ep = 0
        self.ga = 2
        self.al = 2
        self.stepCount = 0
        ## Init Gui

        self.__initGUI(win)

        # Init environment
        self.robot = crawler.CrawlingRobot(self.canvas)
        self.robotEnvironment = crawler.CrawlingRobotEnvironment(self.robot)

        # Init Agent
        actionFn = lambda state: \
          self.robotEnvironment.getPossibleActions(state)
        self.learner = qlearningAgents.QLearningAgent(actionFn=actionFn)

        self.learner.setEpsilon(self.epsilon)
        self.learner.setLearningRate(self.alpha)
        self.learner.setDiscount(self.gamma)

        # Start GUI
        self.running = True
        self.stopped = False
        self.stepsToSkip = 0
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def exit(self):
        self.running = False
        for i in range(5):
            if not self.stopped:
                time.sleep(0.1)
        try:
            self.win.destroy()
        except:
            pass
        sys.exit(0)

    def step(self):
        self.stepCount += 1
        if self.robotEnvironment.isTerminal():
            self.robotEnvironment.reset()
            print('Reset!')
        state = self.robotEnvironment.getCurrentState()
        action = self.learner.getAction(state)
        if action == None:
            raise Exception('Error: no action returned.')
        nextState, reward = self.robotEnvironment.doAction(action)
        self.learner.observeTransition(state, action, nextState, reward)

    def run(self):
        self.stepCount = 0
        self.learner.startEpisode()
        while True:
            minSleep = .01
            tm = max(minSleep, self.tickTime)
            time.sleep(tm)
            self.stepsToSkip = int(tm / self.tickTime) - 1

            if not self.running:
                self.stopped = True
                return
            for i in range(self.stepsToSkip):
                self.step()
            self.stepsToSkip = 0
            self.step()
        self.learner.stopEpisode()

    def start(self):
        self.win.mainloop()

def run():
    global root
    root = tkinter.Tk()
    root.title( 'Crawler GUI' )
    root.resizable( 0, 0 )
    app = Application(root)
    def update_gui():
        app.robot.draw(app.stepCount, app.tickTime)
        root.after(10, update_gui)
    update_gui()

    root.protocol( 'WM_DELETE_WINDOW', app.exit)
    try:
        app.start()
    except:
        app.exit()
