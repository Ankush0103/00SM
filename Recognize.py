import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ankush', 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Leo Messi', 'Madonna']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml') # Using the trained yml fie from face_train.py


img= cv.imread(r"D:\00SM\Facetv\val\ankush\AnkushV6.jpeg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect face in image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 8)
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)        
    cv.putText(img, str(people[label]), (40, 40), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow('Detected image', img)
cv.waitKey(0)










# faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# cv.imshow('Person', frame)
# print(f'Label = {people[label]} with confidence of {confidence}')


# if(confidence<confidence_threshold):
    #     cv.putText(img, "Unknown", (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    #     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    # else:
        # print(f'Label = {people[label]} with confidence of {confidence}')

# confidence_threshold = 0
