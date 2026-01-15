# File with the basic operation of facial recognition

import cv2 
import numpy as np
import face_recognition

# Load image 
messi_image = face_recognition.load_image_file('Images/messi_image1.jpg')
# Convert image to RGB
messi_image = cv2.cvtColor(messi_image, cv2.COLOR_BGR2RGB)

test_image = face_recognition.load_image_file('Images/messi_test_image.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)    

# Find the face on the image
face_location = face_recognition.face_locations(messi_image)[0]
# Encode the face
encode_messi = face_recognition.face_encodings(messi_image)[0]
# Show the face location
cv2.rectangle(messi_image,(face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 0), 2)

# Find the face on the image
face_location_test = face_recognition.face_locations(test_image)[0]
# Encode the face
encode_test = face_recognition.face_encodings(test_image)[0]
# Show the face location
cv2.rectangle(test_image,(face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (0, 255, 0), 2)

# Compare the measurements of the two faces 
results = face_recognition.compare_faces([encode_messi], encode_test) 

# Calculate similarities between faces
face_distance = face_recognition.face_distance([encode_messi], encode_test)
# Print the results in the image
cv2.putText(test_image, f'Result: {results[0]} Similarity: {round(face_distance[0],2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display image
cv2.imshow("Messi", messi_image)
cv2.imshow("Test Image", test_image)
cv2.waitKey(0)
