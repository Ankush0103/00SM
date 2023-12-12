import os
import cv2 as cv
import numpy as np

people = ['Ankush', 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Leo Messi', 'Madonna']
DIR = r'D:\00SM\Facetv\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array  = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print('Training done')

features = np.array(features, dtype='object')
labels = np.array(labels)

# face_recognizer as an instance of LBPHFaceRecognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and labels list
face_recognizer.train(features, labels)


# print(f'Length of features = {len(features)}')
# print(f'Length of labels = {len(labels)}')

# OpenCv allows us to save the train model so that we can use it 
# in another file, directory by using that particular yml file.
face_recognizer.save('face_trained.yml')
np.save('features.npy', features) 
np.save('labels.npy', labels)