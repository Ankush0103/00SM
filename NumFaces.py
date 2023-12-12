import cv2
# Opens a video capture object
cap = cv2.VideoCapture(0) # 0 for internal webcam

while True:
    # Reading a frame from the video stream
    ret, frame = cap.read()
    # Display the processed frame
    cv2.imshow('Video Frame', frame)
    # Press 's' to exit
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
# Releasing the video capture object
cap.release()
cv2.destroyAllWindows()


import cv2 as cv
img = cv.imread('vk1.jpeg') 
# Converting into gray image for better image processing
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
haar_cascade = cv.CascadeClassifier('haar_face.xml')
# Detecting face in image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 3)

for(x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
cv.imshow('Detected Faces', img)
cv.waitKey(0)


print(f'Number of faces -> {len(faces_rect)}') 

# In a group of 5 detects 7 images since haar cascade is sensitive
# to noise, if there is something that looks like a face it will detect that also.
# cv.imshow('Group', img)
# We use varaibles like scaleFactor and minNeogj to detect face and return rectangular coordinates of 
# that face as a list to faces on the score rect
# minNeighbors less, more faces, minimizing variables make haarcascade more prone to noise.
# Haarcascade are more popular but less effective