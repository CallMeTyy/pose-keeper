from transitions import Machine
import collections
import math
import numpy as np
import random


# Think Component: Decision Making
# Things you need to improve: Add states and transitions according to your intervention/rehabilitation coaching design

class Think(object):

    def __init__(self, act, screensize, num_positions=0, alpha=0.2):
        """
        Initializes the state machine and sets up the transition logic.
        :param act_component: Reference to the Act component to trigger visual feedback
        :param flexion_threshold: threshold for entering the flexion state
        :param extension_threshold: threshold for entering the extension state
        """
        self.alpha = alpha  # Smoothing factor
        self.num_positions = num_positions  # Number of positions to smooth
        self.last_positions = [None] * num_positions  # Initialize last positions for each

        self.ball_size = 5
        self.default_ball_pos = np.array((screensize[0]/2, screensize[1]/3*2))
        self.max_ball_size = min(screensize) / 6
        self.ball_pos = self.default_ball_pos
        self.target_pos = np.array((screensize[0]/2, screensize[1]/2))
        self.screensize = screensize
        self.act = act
        self.reset_ball()


    def smooth(self, current_positions):
        smoothed_positions = []

        for i in range(self.num_positions):
            if self.last_positions[i] is None:
                # Initialize with the first position if None
                self.last_positions[i] = current_positions[i]
            else:
                # Apply the EMA formula
                self.last_positions[i] = (
                    self.alpha * current_positions[i] + (1 - self.alpha) * self.last_positions[i]
                )
            smoothed_positions.append(self.last_positions[i])

        return smoothed_positions

    def circle_circle(self, circleA, radiusA, circleB, radiusB):
        return self.distance(circleA,circleB) <= (radiusA+radiusB)
    
    def distance(self,vecA, vecB):
        return math.sqrt((vecA[0]-vecB[0])**2 + (vecA[1]-vecB[1])**2)
    
    def reset_ball(self):
        self.ball_pos = np.array((self.screensize[0]/2, self.screensize[1]/3*2))
        t_x = random.uniform(self.max_ball_size/2, self.screensize[0]-self.max_ball_size/2)
        t_y = random.uniform(self.max_ball_size/2, self.screensize[1]-self.max_ball_size/2)
        self.target_pos = np.array((t_x, t_y))
        self.act.update_target(self.target_pos)
        self.ball_size = 5
    
    def move_ball(self, speed=0.05):
        self.ball_size += speed * self.ball_size
        dist = self.distance(self.ball_pos, self.target_pos)
        if (dist > 2):
            self.ball_pos += (self.target_pos - self.ball_pos)/np.linalg.norm(self.target_pos - self.ball_pos) * 4 * (0.9+dist/self.screensize[1]*3) * (0.8+speed) 
        check_screen = self.max_ball_size - self.ball_size < 0.2
        moved_too_far = self.ball_size > self.max_ball_size + 10
        self.act.update_ball(self.ball_pos, self.ball_size, moved_too_far)
        if (moved_too_far):
            self.reset_ball()
        return check_screen
    
    def touching_ball(self, pos, radius):
        inverted_pos = (self.screensize[0]-int(pos[0]),int(pos[1]))
        return self.circle_circle(inverted_pos,radius,self.target_pos,self.max_ball_size)
    



