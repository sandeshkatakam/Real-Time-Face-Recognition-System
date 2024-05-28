import cv2
import tensorflow as tf
import numpy as np
from preprocessing import Preprocessor


class FaceRecognizer():
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def preprocess_image(self, img):
        img = cv2.resize(img, (160, 160))  # Resize image to the size expected by the FaceNet model
        img = img.astype('float32') / 255  # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def get_embedding(self, img):
        img = self.preprocess_image(img)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(output_details[0]['index'])



from tensorflow.keras.models import load_model

class FaceRecognizerH5():
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, img):
        img = cv2.resize(img, (160, 160))  # Resize image to the size expected by the FaceNet model
        img = img.astype('float32') / 255  # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def get_embedding(self, img):
        img = self.preprocess_image(img)
        return self.model.predict(img)