import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
import tensorflow as tf
import numpy as np
import numpy as np
from scipy import ndimage
from scipy.spatial import distance


class Preprocessor():
    def __init__(self, transforms, width, height, channels):
        super().__init__()
        self.transforms = transforms
        self.width = width
        self.height = height
        self.channels = channels
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  


    def face_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            image = image[y:y+h, x:x+w]
        return image
    
    def face_align(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            eyes = self.eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            if len(eyes) >= 2:
                # Find the largest two eye regions
                eyes = sorted(eyes, key=lambda x: -x[2])[:2]
                # Sort the eye regions such that left_eye[0] <= right_eye[0]
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye, right_eye = eyes
                # Calculate the angle between the two eyes
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dy, dx))
                # Rotate the image to align the eyes horizontally
                M = cv2.getRotationMatrix2D(((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2), angle, 1)
                image = cv2.warpAffine(image, M, (self.width, self.height))
        return image

    def crop_face_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            image = image[y:y+h, x:x+w]
        return image

    def scale_face_image(self, image):
        return cv2.resize(image, (self.width, self.height))

    def normalize_color(self, image):
        return image / 255.0