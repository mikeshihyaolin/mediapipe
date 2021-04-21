# coding=utf-8
"""
@Author: Mikelin
@Contact: mike.lin@ieee.org
@File: hands.py
@Time: 2021-04-20 23:00
@Last_update: 2021-04-20 23:00
"""
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

with mp_hands.Hands(
	static_image_mode=True,
	max_num_hands=2,
	min_detection_confidence=0.7) as hands:
	while cap.isOpened():
		ret, frame = cap.read()	
		results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1))
		image_hight, image_width, _ = frame.shape
		print(results.multi_handedness)
		annotated_image = cv2.flip(frame.copy(), 1)
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks(
					annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
		cv2.imshow('Mediapipe Feed', cv2.flip(annotated_image,1))
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()