




import cv2
import csv
import datetime

# Loading the pre-trained Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading the pre-trained LBPH face recognizer
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

face_recognizer.read('face_trained.yml') # Using the trained yml fie from face_train.py

people = ['Ankush', 'Ben Afflek', 'Elton John', 'Jerry Seinfeld', 'Leo Messi', 'Madonna']

# Setting a confidence threshold for recognition
confidence_threshold = 50  # Adjusting this threshold as needed

cap = cv2.VideoCapture(0)  # 0 for internal webcam and 1 for external webcam

# Creating  a CSV file in append mode to store attendance 
with open('AttendanceCam2.csv', mode='a', newline='') as attendance_file:
    fieldnames = ['Name', ' Date & Time']
    writer = csv.DictWriter(attendance_file, fieldnames=fieldnames)
    recognized_names = []  # Maintain a list of recognized names

    # Writing the header row if the file is empty
    if attendance_file.tell() == 0:
        writer.writeheader()

    while True:      
        ret, frame = cap.read()
        if not ret:
            break
        # Converting the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)  #1.3, 4
        for (x, y, w, h) in faces_rect:
            # Extracting the face region using Region of Interest.
            face_roi = gray[y:y+h, x:x+w]
            # Recognize the face
            label, confidence = face_recognizer.predict(face_roi)
            # Check if the confidence is below the threshold
            if confidence <= confidence_threshold:
                name = 'Unkssnown'
            else:
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

        cv2.imshow('Live Attendance2', frame)

        # Press 's' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

# Releasing the camera and closing all OpenCV windows
cap.release()
cv2.destroyAllWindows()


















        # Detecting faces in the frame
        # Read a frame from the camera
        # if name != 'Unknown':
            #     # Get the current time
            #     current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            #     # Record attendance to CSV
            #     writer.writerow({'Name': name, 'Time': current_time})
            #     attendance_file.flush()

            #     capture_image = True  # Set the flag to stop capturing more images

        # Display the frame with face recognition