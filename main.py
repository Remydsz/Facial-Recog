import face_recognition
import cv2
import numpy as np
import os

video_path = 'tester.mp4'
cap = cv2.VideoCapture(video_path)

# width and height
fw = int(cap.get(3))
fh = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (fw), (fh))

known_faces = []
known_names = []


