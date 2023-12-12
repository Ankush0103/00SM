import cv2
import csv
import datetime

# Loading the pre-trained Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading the pre-trained LBPH face recognizer
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

face_recognizer.read('face_trained.yml') # Using the trained yml fie from face_train.py

# Creating a list of people's names
people = ['Ankush', 'Ben Afflek', 'Elton John', 'Jerry Seinfeld', 'Leo Messi', 'Madonna']

#confidence_threshold = 90

cap = cv2.VideoCapture(0)  # 0 for internal webcam and 1 for external webcam

# Creating  a CSV file in append mode to store attendance 
with open('Attendance1.csv', mode='a', newline='') as attendance_file:
    fieldnames = ['Name', ' Date & Time']
    writer = csv.DictWriter(attendance_file, fieldnames=fieldnames)
    
    # Writing the header row if the file is empty
    if attendance_file.tell() == 0:
        writer.writeheader()

    recognized_names = []  

    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            break
        # Converting the frame to grayscale for better image processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in the frame
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces_rect:
            # Extracting the face region of interest
            face_roi = gray[y:y+h, x:x+w]

            # Recognizing the face
            label, confidence = face_recognizer.predict(face_roi)

#            if confidence>=confidence_threshold:
#                name = "Unknown"

            # else:    
            # Display the name of the recognized person
            name = people[label]

            # Check if the name has already been recorded
            if name not in recognized_names:
                recognized_names.append(name)  # Adding the name to the list of recognized names

                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Getting current time using datetime
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Recording attendance to CSV
                writer.writerow({'Name': name, ' Date & Time': current_time})
                attendance_file.flush()
            
            if name in recognized_names:
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
       
        cv2.imshow('Live Attendance', frame)

        # Press 's' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

# Releasing the camera and closing all OpenCV windows
cap.release()
cv2.destroyAllWindows()

























# Read a frame from the camera