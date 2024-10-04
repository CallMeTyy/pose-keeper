import cv2
import mediapipe as mp
import math
import numpy as np


# Sense Component: Detect joints using the camera
# Things you need to improve: Make the skeleton tracking smoother and robust to errors.
class Sense:

    def __init__(self):
        # Initialize the Mediapipe Pose object to track joints

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()

        # used later for having a moving avergage
        self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.previous_angle = -1

    def detect_joints(self, frame):
        results = self.mp_pose.process(frame)
        return results if results else None

    def calculate_angle(self, joint1, joint2, joint3):
        """
        Calculates the angle between three joints.

        Parameters:
        - joint1: Tuple of (x, y) for the first joint (e.g., shoulder)
        - joint2: Tuple of (x, y) for the middle joint (e.g., elbow)
        - joint3: Tuple of (x, y) for the last joint (e.g., wrist)

        Returns:
        - Angle in degrees between the three joints
        """
        # Calculate vectors
        vector1 = [joint1[0] - joint2[0], joint1[1] - joint2[1]]
        vector2 = [joint3[0] - joint2[0], joint3[1] - joint2[1]]

        # Calculate the dot product and magnitude of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate the angle in radians and convert to degrees
        angle = math.acos(dot_product / (magnitude1 * magnitude2))

        # get a moving average
        self.angle_window.pop(0)
        self.angle_window.append(angle)
        # Use np.convolve to calculate the moving average
        window_size = 10
        angle_mvg = np.convolve(np.asarray(self.angle_window), np.ones(window_size) / window_size, mode='valid')

        return math.degrees(angle_mvg)

    def extract_joint_coordinates(self, landmarks, screensize, joint):
        """
        Extracts the (x, y) coordinates of a specific joint.

        Parameters:
        - landmarks: The list of pose landmarks from MediaPipe
        - joint: The name of the joint (e.g., 'left_elbow')

        Returns:
        - A tuple of (x, y) coordinates of the specified joint
        """
        joint_index_map = {
            'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_INDEX,
            'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
            'left_ankle': mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
            
        }

        joint_index = joint_index_map[joint]
        landmark = landmarks.landmark[joint_index]
        landmark.x = max(min(landmark.x*(screensize[0]-0.2)*1.25,screensize[0]),0)
        landmark.y = max(min(landmark.y*(screensize[1]-0.2)*1.25,screensize[1]),0)
        return (landmark.x, landmark.y), landmark.visibility

    ### Example for defining a function that extracts an angle
    def extract_hip_angle(self, landmarks):
        """
                Extracts the hip angle.

                Parameters:
                - landmarks: The list of pose landmarks from MediaPipe

                Returns:
                - An angle in degrees for the hip
                """
        # extract the x and y coordinates
        left_hip = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_knee = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        return self.calculate_angle(left_shoulder, left_hip, left_knee)

    # Extracts the angle of the knee by measuring the angle between the left hip, left knee, and left ankle
    def extract_knee_angle(self, landmarks):
        # extract the x and y coordinates
        left_hip = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
        return self.calculate_angle(left_hip, left_knee, left_ankle)
