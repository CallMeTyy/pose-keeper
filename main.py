import cv2

from coach import Sense
from coach import Think
from coach import Act

import numpy as np


# Main Program Loop
def main():
    """
    Main function to initialize the exercise tracking application.

    This function sets up the webcam feed, initializes the Sense, Think, and Act components,
    and starts the main loop to continuously process frames from the webcam.
    """

    screen_size = (1080, 720)

    # Initialize the components: Sense for input, Think for decision-making, Act for output
    sense = Sense.Sense()
    act = Act.Act(screen_size)
    think = Think.Think(act, screen_size, 4, 0.2)
    think.set_screen(2)     # Initialize the instructions
    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    # Main loop to process video frames
    while cap.isOpened():

        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Sense: Detect joints
        joints = sense.detect_joints(frame)
        landmarks = joints.pose_landmarks

        # If landmarks are detected, calculate the elbow angle
        if landmarks:
            wrist_r_pos, w_r_vis = sense.extract_joint_coordinates(landmarks, screen_size, "right_wrist")
            wrist_l_pos, w_l_vis = sense.extract_joint_coordinates(landmarks, screen_size, "left_wrist")
            foot_r_pos, f_r_vis = sense.extract_joint_coordinates(landmarks, screen_size, "right_ankle")
            foot_l_pos, f_l_vis = sense.extract_joint_coordinates(landmarks, screen_size, "left_ankle")

            positions = np.array([wrist_r_pos, wrist_l_pos, foot_r_pos, foot_l_pos])
            visibility = np.array([w_r_vis, w_l_vis, f_r_vis, f_l_vis])
            smoothed_landmarks = think.smooth(positions)

            think.update_buttons([smoothed_landmarks[0], smoothed_landmarks[1]])

            # touching_hands = think.circle_circle(smoothed_landmarks[0],100,smoothed_landmarks[1],100)

            act.provide_feedback(frame=frame, joints=joints)
            think.update_state(smoothed_landmarks, visibility)

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
