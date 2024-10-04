# Act Component: Provide feedback to the user

import mediapipe as mp
import cv2
import numpy as np
import random
import pyttsx3
import time

# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self, screensize):
        self.engine = pyttsx3.init()
        self.screensize = screensize
        self.prevTime = 0
        self.balloon_size = 50
        self.ball_size = 10
        self.ball_pos = (0,0)
        self.hit_screen = False
        self.target_pos = (0,0)
        self.buttons = []
        self.game_bg = cv2.imread("coach/assets/field.png")
        self.main_menu_bg = cv2.imread("coach/assets/blurred_field.png")
        self.left_hand = cv2.imread("coach/assets/left_hand.png")
        self.right_hand = cv2.imread("coach/assets/right_hand.png")

    def update_ball(self, ball_pos, ball_size, hit_screen):
        self.ball_pos = ball_pos
        self.ball_size = ball_size
        self.hit_screen = hit_screen

    def update_target(self, target_pos):
        self.target_pos = target_pos

    def update_button_list(self, buttons):
        self.buttons = buttons

    def visualize_buttons(self,background):
        for button in self.buttons:
            cv2.circle(background, button.pos, button.size, button.color, -1)
            cv2.circle(background, button.pos, int(button.size * button.progress), (200, 200, 255), -1)
            cv2.putText(background, button.text, button.pos-np.array([button.size,0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)


    def visualize_game(self, smoothed_landmarks,visibility, score, lives):
        """
        Renders the game .
        """
        # Create a black background
        img = self.game_bg  # np.ones((self.screensize[1], self.screensize[0], 3), dtype=np.uint8)
        img = cv2.resize(img, self.screensize, interpolation=cv2.INTER_AREA)


        # Show the ball
        ball_color = (255,255,255) if not self.hit_screen else (0,255,0)
        ball_pos_int = (int(self.ball_pos[0]),int(self.ball_pos[1]))
        ball_rad_int = int(self.ball_size)
        cv2.circle(img, ball_pos_int, ball_rad_int, ball_color, -1)

        # Show the place where the ball will go
        t_pos = (int(self.target_pos[0]), int(self.target_pos[1]))
        cv2.putText(img, f'!', t_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img, f'SCORE: {score}', (int(self.screensize[0]/3),30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'LIVES: {lives}', (int(self.screensize[0] / 3), 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        self.base_scene(img,smoothed_landmarks,visibility)
        # Show the image in the window
        cv2.imshow('Keep the ball from entering the goal!', img)

        # Wait for 1 ms and check if the window should be closed
        cv2.waitKey(1)

    def base_scene(self,img, smoothed_landmarks, visibility):
        fps = 1 / (time.time() - self.prevTime)
        self.prevTime = time.time()
        self.visualize_buttons(img)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 10),cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the limbs
        touching_hands = False

        color = (0, 0, 255) if not touching_hands else (0, 255, 0)
        for landmark_pos, vis in zip(smoothed_landmarks, visibility):
            if vis > 0.4:
                pos = (self.screensize[0] - int(landmark_pos[0]), int(landmark_pos[1]))
                cv2.circle(img, pos, self.balloon_size, color, -1)

    def visualize_main_menu(self, smoothed_landmarks,visibility):
        img = self.main_menu_bg#np.ones((self.screensize[1], self.screensize[0], 3), dtype=np.uint8)
        img = cv2.resize(img, self.screensize,interpolation = cv2.INTER_AREA)
        self.base_scene(img,smoothed_landmarks,visibility)

        # Wait for 1 ms and check if the window should be closed
        cv2.imshow('Keep the ball from entering the goal!', img)
        cv2.waitKey(1)

    def provide_feedback(self, frame, joints):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.
        :param elbow_angle_mvg: The moving average from the left elbow angle.

        """

        mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Define the number and text to display
        # number = elbow_angle_mvg
        # text = " "
        # if decision == 'flexion':
        #     text = "You are flexing your elbow! %s" % number
        # elif decision == 'extension':
        #     text = "You are extending your elbow! %s" % number


        # Set the position, font, size, color, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_color = (0, 0, 0)  # White color in BGR
        thickness = 2

        # Define the position for the number and text
        text_position = (50, 50)

        # Draw the text on the image
        cv2.putText(frame, "gaming", text_position, font, font_scale, font_color, thickness)

        # Display the frame (for debugging purposes)
        cv2.imshow('Sport Coaching Program', frame)
