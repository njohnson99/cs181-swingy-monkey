# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

# should have at most 5 buckets for each feature
# so increase W and H because we're dividing by them
W = 150
H = 100
V = 10
# screw around with epsilon, could start as high as 1 in the extreme case
discount = 0.9
# this is a good learning rate
learning = 0.05
screen_width  = 600
screen_height = 400
decrease_factor = 2

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        #print('ONLY SUPPOSED TO HAPPEN 1 TIME')
        print "new game"
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.game_num = 1
        self.Q = np.zeros((4, 4, 4, 2, 2, 2)) # because there are 2 actions
        self.epsilon = 0.8
        self.gravity = None

    def reset(self):
        self.game_num += 1
        print("game number " + str(self.game_num))
        #print(self.Q)      
        self.epsilon = self.epsilon / decrease_factor
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    def discretize_state(self, state):
        tree_dist = state['tree']['dist'] / W

        gap = ((state['tree']['top'] + state['tree']['bot'])/2 - (state['monkey']['top'] + state['monkey']['bot'])/2) / H

        # gap_bot = (state['monkey']['bot'] - state['tree']['bot']) / H
        # gap_top = (state['tree']['top'] - state['monkey']['top']) / H

        # collapse tree_top, tree_bot, monkey_top, and monkey_bot into 1 feature: midpoint of tree minus midpoint of monkey
        # gap = ((state['tree']['top'] + state['tree']['bot'])/2 - (state['monkey']['top'] + state['monkey']['bot'])/2) / H

        # tree_top = (screen_height - state['tree']['top']) / H
        # tree_bot = (screen_height - state['tree']['bot']) / H
        
        screen_gap = (((state['monkey']['top'] + state['monkey']['bot'])/2) - 200) / H
        # monkey_top = (screen_height - ((state['monkey']['top'] + state['monkey']['top'])/2) / H
        # monkey_bot = state['monkey']['bot'] / H
        
        # boundary_dist = min(monkey_top, monkey_bot)
        #print(boundary_dist)
        
        # bin into groups of velocities
        if state['monkey']['vel'] / V < 0:
            velocity = 0
        else:
            velocity = 1

        # return tree_dist, tree_top, tree_bot, monkey_top, monkey_bot, velocity
        return int(tree_dist), int(gap), int(screen_gap), int(velocity)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        if self.last_state != None and state != None:

            if self.gravity == None:
                if state['monkey']['vel'] == -1:
                    self.gravity = 0
                else:
                    self.gravity = 1

            tree_dist_1, gap_1, screen_gap_1, velocity_1 = self.discretize_state(self.last_state)
            tree_dist_2, gap_2, screen_gap_2, velocity_2 = self.discretize_state(state)

            # new_action = np.argmax(self.Q[tree_dist_2][gap_2][screen_gap_2][velocity_2])

            if np.random.binomial(1, self.epsilon) == 1:
                new_action = np.random.binomial(1, 0.5)
                print("randomly selecting to " + str(new_action))

            else:
                new_action = np.argmax(self.Q[tree_dist_2][gap_2][screen_gap_2][velocity_2][self.gravity])
                if new_action == 1:
                    print self.Q[tree_dist_2][gap_2][screen_gap_2][velocity_2][self.gravity]

            self.Q[tree_dist_1][gap_1][screen_gap_1][velocity_1][self.gravity][self.last_action] = \
            (1 - learning) * self.Q[tree_dist_1][gap_1][screen_gap_1][velocity_1][self.gravity][self.last_action] + \
            learning * (self.last_reward + discount * max(self.Q[tree_dist_2][gap_2][screen_gap_2][velocity_2][self.gravity]))
            
            # self.Q[tree_dist_1][tree_top_1][tree_bot_1][monkey_top_1][monkey_bot_1][velocity_1][self.last_action] = \
            # (1 - learning) * self.Q[tree_dist_1][tree_top_1][tree_bot_1][monkey_top_1][monkey_bot_1][velocity_1][self.last_action] + \
            # learning * (self.last_reward + discount * max(self.Q[tree_dist_2][tree_top_2][tree_bot_2][monkey_top_2][monkey_bot_2][velocity_2]))
            # new_action = np.argmax(self.Q[tree_dist_2][tree_top_2][tree_bot_2][monkey_top_2][monkey_bot_2][velocity_2])
        
        else:
            print state
            tree_dist_2, gap_2, screen_gap_2, velocity_2 = self.discretize_state(state)
            new_action = 0
            # new_action = np.argmax(self.Q[tree_dist_2][gap_2][screen_gap_2][velocity_2])

        new_state  = state
        self.last_action = new_action
        self.last_state  = new_state


        return self.last_action

        # delta between monkey_top and tree_top
        # don't update before you die?

        # turn animation off to run faster
        # 100 might be a good score to shoot for? consistently get 50-100?
        # init Q to 0

        # a lot of it is just adjusting parameters

        # epsilon should fall exponentially over time (annealing)
        # in epsilon-greedy:
        # SARSA will sometimes update the max
        # Q-learning will always update the max
        # this is because your policy is to choose the max with prob 1 - epsilon and explore with prob epsilon
        # there's a subtle diff between SARSA and Q-learning but doesn't really matter?

        # don't focus on gravity first, because it's hard to learn in the high-gravity world 
        # (apparently their team couldn't get it working last year lol)
        # but what you would do is look at the velocity and decide whether it's high gravity or low gravity
        # add gravity as a parameter to your state (but it doesn't help much anyway)

        # use the update equation for w_sa but replace w_sa with Q_sa

        # You might do some learning here based on the current state and the last state.

        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.

    def reward_callback(self, reward):
        #print('my reward is')
        #print(reward)
        '''This gets called so you can see what reward you get.'''
        
        #we need to implement this
        #if i hit 

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 100, 10)

	# Save history. 
	np.save('hist',np.array(hist))


