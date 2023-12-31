
import cv2

# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)  
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h] # Updating coords in our list
    return coords

# Method to detect the features
def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, then draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4: # if no face dtected then len(coords)==0
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = draw_boundary(roi_img, eyeCascade, 1.1, 14, color['red'], "Eye")  #14, 3
        coords = draw_boundary(roi_img, noseCascade, 1.1, 10, color['green'], "Nose") #5
        coords = draw_boundary(roi_img, mouthCascade, 1.1, 25, color['white'], "Mouth") #20
    return img

cap = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    isTrue, img = cap.read()
    img = detect(img, faceCascade, eyesCascade , noseCascade, mouthCascade)
    cv2.imshow("Face detection", img) # Processed
    # Press 's' to exit the loop and close camera
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# releasing webcam and closing all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
