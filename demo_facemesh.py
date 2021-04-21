# coding=utf-8
"""
@Author: Mikelin
@Contact: mike.lin@ieee.org
@File: demo_facemesh.py
@Time: 2021-04-20 23:00
@Last_update: 2021-04-21 11:10
"""
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
	static_image_mode=True,
	max_num_faces=1,
	min_detection_confidence=0.5) as face_mesh:
	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue
		results = face_mesh.process(frame)
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				mp_drawing.draw_landmarks(
				image=frame,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACE_CONNECTIONS,
				landmark_drawing_spec=drawing_spec,
				connection_drawing_spec=drawing_spec)
		cv2.imshow('MediaPipe FaceMesh', frame)

		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()