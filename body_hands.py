# coding=utf-8
"""
@Author: Mikelin
@Contact: mike.lin@ieee.org
@File: body_hands.py
@Time: 2021-04-20 23:00
@Last_update: 2021-04-20 23:00
"""
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) 
cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret, frame = cap.read()
	# Make detection
	res_hand = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1))
	res_body = pose.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1))
	annotated_image = cv2.flip(frame.copy(), 1)
	# Render Detected Hands
	if res_hand.multi_hand_landmarks:
		for hand_landmarks in res_hand.multi_hand_landmarks:
			mp_drawing.draw_landmarks(
				annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
	# Render Detected Pose
	if res_body.pose_landmarks:
		 (annotated_image, res_body.pose_landmarks, mp_pose.POSE_CONNECTIONS,
		mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
		mp_drawing.DrawingSpec(color=(245,66,230), thickness=5, circle_radius=5))	
	cv2.imshow('Mediapipe Feed', cv2.flip(annotated_image,1))
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()