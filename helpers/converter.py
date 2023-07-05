import cv2
import json
import numpy as np
import pywt


class Converter:
    def __init__(self, config_path):
        with open(config_path) as file:
            self.config = json.load(file)
            
        self.face_cascade = cv2.CascadeClassifier(
            "helpers/haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(
            "helpers/haarcascade_eye.xml")

    def extract_facial_features(iself, image, mode='haar', level=1):
        img_array = image
        img_array = np.float32(img_array)

        img_array /= 255

        coeffs = pywt.wavedec2(img_array, mode, level=level)

        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0

        img_array_H = pywt.waverec2(coeffs_H, mode)
        img_array_H *= 255
        img_array_H = np.uint8(img_array_H)

        return img_array_H

    def crop_and_resize(self, frame):

        # Perform face detection
        faces = self.face_cascade.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the detected face region
            face = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(face)
            if len(eyes) >= 2:
                return cv2.resize(face, (self.config['resize_w_h'], self.config['resize_w_h'])), (x, y, w, h)
