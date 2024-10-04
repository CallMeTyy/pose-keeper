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
        self.left_hand = cv2.imread("coach/assets/left_hand.png",cv2.IMREAD_UNCHANGED)
        self.right_hand = cv2.imread("coach/assets/right_hand.png",cv2.IMREAD_UNCHANGED)
        self.left_foot = cv2.imread("coach/assets/left_hand.png",cv2.IMREAD_UNCHANGED)
        self.right_foot = cv2.imread("coach/assets/right_hand.png",cv2.IMREAD_UNCHANGED)
        self.leaderboard = cv2.imread("coach/assets/leaderboard.png",cv2.IMREAD_UNCHANGED)
        self.graphics = [self.right_hand, self.left_hand, self.right_foot, self.left_foot]
        self.ball_graphic = cv2.imread("coach/assets/ball.png",cv2.IMREAD_UNCHANGED)
        self.target_graphics = cv2.imread("coach/assets/transparent_gloves.png",cv2.IMREAD_UNCHANGED)

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
            if not button.image:
                cv2.circle(background, button.pos, button.size, button.color, -1)
                cv2.circle(background, button.pos, int(button.size * button.progress), (200, 200, 255), -1)
                self.put_centered_text(background, button.text, button.pos, cv2.FONT_HERSHEY_SIMPLEX, 7/len(button.text), (0,0,0), 2)
            else:
                gfx = self.change_transparency(button.image_graphic, 0.4+button.progress*0.6)
                self.place_image_on_top(background,gfx,button.pos)



    def visualize_game(self, smoothed_landmarks,visibility, score, lives):
        """
        Renders the game .
        """
        # Create a black background
        img = self.game_bg  # np.ones((self.screensize[1], self.screensize[0], 3), dtype=np.uint8)
        img = cv2.resize(img, self.screensize, interpolation=cv2.INTER_AREA)


        # Show the ball
        #ball_color = (255,255,255) if not self.hit_screen else (0,255,0)
        ball_pos_int = (int(self.ball_pos[0]),int(self.ball_pos[1]))
        ball_rad_int = max(int(self.ball_size),1)

        ball = cv2.resize(self.ball_graphic, (ball_rad_int,ball_rad_int))
        ball = self.rotate_image(ball,np.sin(self.ball_size * 10)*180)
        self.place_image_on_top(img,ball,ball_pos_int)
        #cv2.circle(img, ball_pos_int, ball_rad_int, ball_color, -1)

        # Show the place where the ball will go
        if score <= 10:
            t_pos = (int(self.target_pos[0]), int(self.target_pos[1]))
            graphic = self.change_transparency(self.target_graphics, 1-score/10)
            self.place_image_on_top(img, graphic, t_pos)
        
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

        #color = (0, 0, 255) if not touching_hands else (0, 255, 0)
        for i in range(len(smoothed_landmarks)):
            vis = visibility[i]
            landmark_pos = smoothed_landmarks[i]
            if vis > 0.4 or i <= 1:
                pos = (self.screensize[0] - int(landmark_pos[0]), int(landmark_pos[1]))
                self.place_image_on_top(img, self.graphics[i], pos)
                #cv2.circle(img, pos, self.balloon_size, color, -1)


    def visualize_main_menu(self, smoothed_landmarks,visibility,score):
        img = self.main_menu_bg#np.ones((self.screensize[1], self.screensize[0], 3), dtype=np.uint8)
        img = cv2.resize(img, self.screensize,interpolation = cv2.INTER_AREA)
        self.place_image_on_top(img, self.leaderboard, (int(self.screensize[0] / 6), int(self.screensize[1] / 4 * 3)))

        self.put_centered_text(img, f'Hover over the start button to start!',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 8*7)),font_scale=0.8)

        self.base_scene(img,smoothed_landmarks,visibility)

        self.put_centered_text(img, f'POSEKEEPER',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 5)))



        if score > 0:
            self.put_centered_text(img, f'YOUR SCORE WAS: {score}', (int(self.screensize[0]/2),int(self.screensize[1]/3))
        , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        # Wait for 1 ms and check if the window should be closed
        cv2.imshow('Keep the ball from entering the goal!', img)
        cv2.waitKey(1)

    def visualize_instructions(self, smoothed_landmarks,visibility):
        img = self.main_menu_bg
        img = cv2.resize(img, self.screensize,interpolation = cv2.INTER_AREA)

        self.base_scene(img,smoothed_landmarks,visibility)

        self.put_centered_text(img, f'Welcome to POSEKEEPER',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 8)),color=(0,0,0),font_scale=1.5,thickness=6)

        self.put_centered_text(img, f'Stand at least 1.5 meter from the camera.',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 8*2)),color=(0,0,0))

        self.put_centered_text(img, f'Catch the ball!',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 8 * 5)),font_scale=1.5,thickness=5)
        self.put_centered_text(img, f'Try moving both hands to the glove outlines',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 10*7)),font_scale=0.8)
        self.put_centered_text(img, f'Hover over buttons to interact!',
                               (int(self.screensize[0] / 2), int(self.screensize[1] / 11 * 10)), font_scale=0.8)

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

    @staticmethod
    def rotate_image(image, angle):
        # Get the dimensions of the image
        (h, w) = image.shape[:2]

        # Calculate the center of the image
        center = (w // 2, h // 2)

        # Get the rotation matrix (rotating by the given angle around the center)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Perform the rotation using warpAffine
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        return rotated_image

    @staticmethod
    def place_image_on_top(base_img, overlay_img, position):
        # Get dimensions of the base image and overlay image
        base_h, base_w = base_img.shape[:2]
        overlay_h, overlay_w = overlay_img.shape[:2]

        # Calculate top-left corner for the overlay image to be centered
        x_center, y_center = position
        x1 = int(x_center - overlay_w // 2)
        y1 = int(y_center - overlay_h // 2)

        # Make sure the overlay image doesn't go out of bounds
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x1 + overlay_w > base_w: x1 = base_w - overlay_w
        if y1 + overlay_h > base_h: y1 = base_h - overlay_h

        # Extract the region of interest (ROI) from the base image where the overlay will be placed
        roi = base_img[y1:y1 + overlay_h, x1:x1 + overlay_w]

        # If overlay image has transparency, we need to handle alpha blending
        if overlay_img.shape[2] == 4:  # Check if the overlay has an alpha channel
            overlay_rgb = overlay_img[:, :, :3]
            alpha = overlay_img[:, :, 3] / 255.0  # Normalize the alpha channel

            # Blend the overlay with the ROI using alpha blending
            for c in range(0, 3):
                roi[:, :, c] = (1.0 - alpha) * roi[:, :, c] + alpha * overlay_rgb[:, :, c]
        else:
            # If there's no alpha channel, simply overwrite the ROI with the overlay
            base_img[y1:y1 + overlay_h, x1:x1 + overlay_w] = overlay_img

        return base_img

    @staticmethod
    def put_centered_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255),
                          thickness=2):
        # Get the text size (width, height) and the baseline
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # Calculate the bottom-left corner for the text to be centered at the given position
        x = position[0] - text_width // 2
        y = position[1] + text_height // 2

        # Place the text on the image
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        return image

    @staticmethod
    def change_transparency(image, alpha_value):
        """
        Change the transparency of an image by adjusting the alpha channel.

        Parameters:
        - image: Input image with an alpha channel (RGBA).
        - alpha_value: A float between 0.0 (fully transparent) and 1.0 (fully opaque).

        Returns:
        - The image with adjusted transparency.
        """
        if image.shape[2] == 4:  # Ensure the image has an alpha channel
            # Extract the RGB channels and the alpha channel
            rgb_channels = image[:, :, :3]
            alpha_channel = image[:, :, 3]

            # Adjust the alpha channel based on the provided alpha_value
            adjusted_alpha = (alpha_channel * alpha_value).astype(np.uint8)

            # Merge the RGB channels with the adjusted alpha channel
            transparent_image = np.dstack([rgb_channels, adjusted_alpha])

            return transparent_image
        else:
            raise ValueError("The image does not have an alpha channel (it must be RGBA).")
