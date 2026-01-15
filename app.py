import cv2 
import numpy as np
import face_recognition
import os 
from datetime import datetime

# Get folder of images
path = 'Images'
images = []
classNames = []
list_images = os.listdir(path)

for cl in list_images:
    current_image = cv2.imread(f'{path}/{cl}')
    # Add image to list of images
    images.append(current_image)
    # Get the name without the extension
    classNames.append(os.path.splitext(cl)[0])

# Function to encode images
def encoding_images(images):
    encoded_images = []
    for img in images:
        # Convert the image from BGR to RGB and encode it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # Add encoded images to the list
        encoded_images.append(encode)
    return encoded_images

# Function to identify and mark attendance in the CSV file
def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')

encoded_images = encoding_images(images)

# Start the webcam
capture = cv2.VideoCapture(0)

# Start the video capture
while True:
    # Capture frame-by-frame
    success, img = capture.read()
    # Resize the image and convert
    small_image = cv2.resize(img,(0,0),None,0.25,0.25)
    small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
    # Find faces and encode them
    face_current_frame = face_recognition.face_locations(small_image)
    encoded_current_frame = face_recognition.face_encodings(small_image, face_current_frame)

    for encode_face, face_location in zip(encoded_current_frame, face_current_frame):
        # Compare faces
        matches = face_recognition.compare_faces(encoded_images, encode_face)
        face_distance = face_recognition.face_distance(encoded_images, encode_face)
        print(face_distance)
        # Get the index of the closest match
        matchIndex = np.argmin(face_distance)

        # If a match is found, get the name and distance
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"Matched {name} with distance {face_distance[matchIndex]}")
            # Get the bounding box coordinates
            y1, x2, y2, x1 = face_location
            # Resize the bounding box (0.25 to 1 scale)
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img,(x1, y2-35),(x2, y2),(0,255,0),cv2.FILLED)
            # Show the name 
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    # Display camera frame
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)