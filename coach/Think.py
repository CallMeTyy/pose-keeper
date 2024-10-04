from transitions import Machine
import collections
import math
import numpy as np
import random
import time


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
        self.buttons = []
        self.state = 0
        self.score = 0
        self.max_lives = 5
        self.lives = 5


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
    @staticmethod
    def circle_circle(circleA, radiusA, circleB, radiusB):
        dist = math.sqrt((circleA[0]-circleB[0])**2 + (circleA[1]-circleB[1])**2)
        return dist <= (radiusA+radiusB)
    @staticmethod
    def distance(vecA, vecB):
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
            self.ball_pos += (self.target_pos - self.ball_pos)/np.linalg.norm(self.target_pos - self.ball_pos) * 4 * (0.9+dist/self.screensize[1]*3) * (25*speed)
        check_screen = self.max_ball_size - self.ball_size < 0.2
        moved_too_far = self.ball_size > self.max_ball_size + 10
        self.act.update_ball(self.ball_pos, self.ball_size, moved_too_far)
        return check_screen
    
    def touching_ball(self, pos, radius):
        inverted_pos = (self.screensize[0]-int(pos[0]),int(pos[1]))
        return self.circle_circle(inverted_pos,radius,self.target_pos,self.max_ball_size)

    def update_buttons(self, hand_positions):
        for button in self.buttons:
            button.run(hand_positions, True)

    def remove_buttons(self):
        self.buttons.clear()

    def add_button(self,function, position,size,text="Button",color=(230,230,230)):
        b = Button(function, position, size,self.screensize,text,color)
        self.buttons.append(b)
        self.act.update_button_list(self.buttons)

    def debug_print(self,text="Gaming"):
        print(text)

    def check_ball_collision(self, smoothed_landmarks, visibility):
        # Holding hands together
        touching_hands = self.circle_circle(smoothed_landmarks[0], 100, smoothed_landmarks[1], 100)
        # Check if hands is touching ball
        catch_ball = False
        if touching_hands:
            catch_ball = self.touching_ball(smoothed_landmarks[0], 100) or self.touching_ball(smoothed_landmarks[1], 100)
            print(catch_ball)
        # Check if right feet is touching ball
        if visibility[2] > 0.1 and not catch_ball:
            catch_ball = self.touching_ball(smoothed_landmarks[2], 50)
        # Check if left feet is touching ball
        if visibility[3] > 0.1 and not catch_ball:
            catch_ball = self.touching_ball(smoothed_landmarks[3], 50)
        return catch_ball

    def set_screen(self, state):
        self.remove_buttons()
        if state == 0:
            self.add_button(self.start_game, (int(self.screensize[0]/2), int(self.screensize[1]/3*2)),100, "Start")
        elif state == 1:
            self.score = 0
            self.lives = self.max_lives
            self.add_button(self.main_menu, (50, 50), 50, "Exit")

        self.state = state

    def update_state(self, smoothed_landmarks, visibility):
        if self.state == 0:
            self.act.visualize_main_menu(smoothed_landmarks, visibility,self.score)

        elif self.state == 1:
            if self.move_ball(0.025 + self.score * 0.0025):
                if self.check_ball_collision(smoothed_landmarks, visibility):
                    self.score += 1
                    self.reset_ball()
                else:
                    print("MISS")
                    self.lives -= 1
                    self.reset_ball()
            if self.lives >= 1:
                self.act.visualize_game(smoothed_landmarks, visibility,self.score,self.lives)
            else:
                self.main_menu()

    def start_game(self):
        self.set_screen(1)

    def main_menu(self):
        self.set_screen(0)


class Button(object):
    def __init__(self, function, position, size, screen_size, text = "Button", color = (255,255,255)):
        self.pos = position
        self.size = size
        self.screensize = screen_size
        self.is_colliding = False
        self.interaction_time = 1
        self.interaction_start_time = 0
        self.text = text
        self.color = color
        self.progress = 0
        self.function = function

    def test_collision(self, other, other_rad = 25):
        return Think.circle_circle(self.pos,self.size, other, other_rad)

    def run(self, hand_positions, invert_hands = False):
        r_hand = (int(hand_positions[0][0]), int(hand_positions[0][1]))
        l_hand = (int(hand_positions[1][0]), int(hand_positions[1][1]))
        if invert_hands:
            r_hand = (self.screensize[0]-r_hand[0],r_hand[1])
            l_hand = (self.screensize[0]-l_hand[0],l_hand[1])

        if self.test_collision(r_hand) or self.test_collision(l_hand):
            if not self.is_colliding:
                self.is_colliding = True
                self.interaction_start_time = time.time()
            else:
                self.progress = (time.time() - self.interaction_start_time) / self.interaction_time
                if time.time() - self.interaction_start_time > self.interaction_time:
                    self.function()

        elif self.is_colliding:
            self.is_colliding = False
            self.progress = 0



