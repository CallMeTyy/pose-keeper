from transitions import Machine
import collections
import math
import numpy as np
import random
import time
import vlc
import cv2
import sys


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
        self.score = 0
        self.ball_size = 5
        self.default_ball_pos = np.array((screensize[0]/2, screensize[1]/3*2))
        self.max_ball_size = min(screensize) / 6
        self.ball_pos = self.default_ball_pos
        self.target_pos = np.array((screensize[0]/2, screensize[1]/2))
        self.screensize = screensize
        self.act = act
        self.reset_ball()
        self.buttons = []
        self.state = 2

        self.max_lives = 5
        self.lives = 5
        self.is_kicked = False
        self.ambience = vlc.MediaPlayer("coach/assets/ambience.wav")
        self.catch_sound = vlc.MediaPlayer("coach/assets/catch.mp3")
        self.kick_sound = vlc.MediaPlayer("coach/assets/kick.wav")
        self.cheer_sound = vlc.MediaPlayer("coach/assets/cheer.wav")
        self.cheer_sound.audio_set_volume(60)
        self.miss_sound = vlc.MediaPlayer("coach/assets/aww.wav")
        self.mot1_sound = vlc.MediaPlayer("coach/assets/mot1.mp3")
        self.mot2_sound = vlc.MediaPlayer("coach/assets/mot2.mp3")
        self.mot3_sound = vlc.MediaPlayer("coach/assets/mot3.mp3")
        self.bad1_sound = vlc.MediaPlayer("coach/assets/bad1.mp3")
        self.bad2_sound = vlc.MediaPlayer("coach/assets/bad2.mp3")
        self.bad3_sound = vlc.MediaPlayer("coach/assets/bad3.mp3")
        self.mot_sounds = [self.mot1_sound, self.mot2_sound,self.mot3_sound]
        self.bad_sounds = [self.bad1_sound, self.bad2_sound,self.bad3_sound]


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
        self.ball_size = -10
        self.is_kicked = False
    
    def move_ball(self, speed=0.05):
        self.ball_size += speed * max(self.ball_size,10)
        if self.ball_size > 0 and not self.is_kicked:
            self.is_kicked = True
            self.play_reset(self.kick_sound)
        dist = self.distance(self.ball_pos, self.target_pos)
        if dist > 5 and self.ball_size > 0:
            self.ball_pos += (self.target_pos - self.ball_pos)/np.linalg.norm(self.target_pos - self.ball_pos) * 4 * (0.9+dist/self.screensize[1]*3) * (25*speed)
        else:
            self.ball_pos = self.target_pos
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

    def add_button(self,function, position,size,text="Button",color=(230,230,230),image="",both_hands=False,interaction_time = 1):
        b = Button(function, position, size,self.screensize,text,color,image=image,both_hands=both_hands)
        b.interaction_time = interaction_time
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
            catch_ball = self.touching_ball(smoothed_landmarks[0], 120) or self.touching_ball(smoothed_landmarks[1], 120)
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
            self.add_button(self.start_game, (int(self.screensize[0]/2), int(self.screensize[1]/3*2)),100, "Start",interaction_time=2)
            self.add_button(self.instructions_menu, (int(self.screensize[0] / 8*7), int(self.screensize[1] / 8)), 60,
                            "Instructions")
            self.add_button(sys.exit, (int(self.screensize[0] / 8), int(self.screensize[1] / 8)), 60,
                            "Exit")
        elif state == 1:
            self.ambience.play()
            self.score = 0
            self.lives = self.max_lives
            #self.add_button(self.main_menu, (50, 50), 50, "Exit")
        elif state==2:
            self.add_button(self.finish_instruction, (int(self.screensize[0] / 2), int(self.screensize[1] / 2)), 150,
                            image="coach/assets/transparent_gloves.png",both_hands=True)

        self.state = state

    def update_state(self, smoothed_landmarks, visibility):
        if self.state == 0:
            self.act.visualize_main_menu(smoothed_landmarks, visibility,self.score)
        elif self.state == 1:
            if self.move_ball(0.025 + self.score * 0.0025):
                if self.check_ball_collision(smoothed_landmarks, visibility):
                    self.score += 1
                    self.reset_ball()
                    self.play_reset(self.catch_sound)
                    self.play_reset(self.cheer_sound)
                    self.play_random_mot()
                else:
                    print("MISS")
                    self.lives -= 1
                    self.reset_ball()
                    self.play_random_bad()
                    self.play_reset(self.miss_sound)
            if self.lives >= 1:
                self.act.visualize_game(smoothed_landmarks, visibility,self.score,self.lives)
            else:
                self.main_menu()
        elif self.state == 2:
            self.act.visualize_instructions(smoothed_landmarks, visibility)
        if self.ambience.get_state() == vlc.State.Ended:
            self.play_reset(self.ambience)

    def start_game(self):
        self.set_screen(1)

    def main_menu(self):
        self.set_screen(0)

    def instructions_menu(self):
        self.set_screen(2)

    def finish_instruction(self):
        self.remove_buttons()
        self.add_button(self.main_menu, (int(self.screensize[0] / 6*5), int(self.screensize[1] / 6*5)), 70,
                            "OK!")

    def play_reset(self,audio):
        if audio.get_state() == vlc.State.Ended:
            audio.stop()
        else:
            audio.pause()
        audio.set_time(0)
        audio.play()

    def play_random_mot(self):
        if random.random() < max(0.2, 0.8-(self.score/50)):
            index = math.floor(random.uniform(0,len(self.mot_sounds)))
            self.play_reset(self.mot_sounds[index])

    def play_random_bad(self):
        if random.random() < max(0.2, 0.8-(self.score/50)):
            index = math.floor(random.uniform(0,len(self.bad_sounds)))
            self.play_reset(self.bad_sounds[index])


class Button(object):
    def __init__(self, function, position, size, screen_size, text = "Button", color = (255,255,255), image = "",both_hands = False):
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
        self.image = image
        self.both_hands = both_hands
        if self.image:
            self.image_graphic = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    def test_collision(self, other, other_rad = 50):
        return Think.circle_circle(self.pos,self.size, other, other_rad)

    def run(self, hand_positions, invert_hands = False):
        r_hand = (int(hand_positions[0][0]), int(hand_positions[0][1]))
        l_hand = (int(hand_positions[1][0]), int(hand_positions[1][1]))
        if invert_hands:
            r_hand = (self.screensize[0]-r_hand[0],r_hand[1])
            l_hand = (self.screensize[0]-l_hand[0],l_hand[1])

        r_hand_collide = self.test_collision(r_hand)
        l_hand_collide = self.test_collision(l_hand)
        if ((r_hand_collide or l_hand_collide) and not self.both_hands) or (r_hand_collide and l_hand_collide and self.both_hands):
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



