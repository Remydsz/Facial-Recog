import face_recognition
import cv2
import numpy as np
import os

print("init")

video_path = 'tester.mp4'
cap = cv2.VideoCapture(video_path)

# width and height
fw = int(cap.get(3))
fh = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (fw), (fh), True)

known_faces = []
known_names = []

osp = 'Macintosh HD\\Users\\remyinsync\\Downloads\\Face Cog'
if os.path.exists(osp):
    for filename in os.listdir(osp):
        if filename.endswith('.jpg'):
            print(f"Processing {filename}")
            image_path = os.path.join(osp, filename)
            image = face_recognition.load_image_file(image_path)

            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                face_encoding = encodings[0]
                known_faces.append(face_encoding)
                known_names.append("no bitches")
            else:
                print(f"No faces found in {filename}")

print(f"Loaded {len(known_faces)} encodings.")

face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video input.")
        break
    if process_this_frame:
        print("Processing frame...")

        # optional resizing for optimization
        sf = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        sf = frame

        # Find faces and face encodings
        face_locations = face_recognition.face_locations(sf)
        print(f"Found {len(face_locations)} face(s).")

        face_encodings = face_recognition.face_encodings(sf, face_encodings)

        # Is face a match for known faces
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_face(
                known_faces, face_encoding, tolerance = 0.80)
            name = "LADJFLDJLFJKS"

            # Known face with the smallest distance to current face
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces[best_match_index]

            face_names.append(name)

        print(f"Identified {face_names}.")

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

    out.write(frame)

out.release()
cap.release()


