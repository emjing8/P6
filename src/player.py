from config import BOARD_SIZE, categories, image_size
from tensorflow.keras import models
import numpy as np
import tensorflow as tf

class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()

class UserInputPlayer:
    def get_move(self, board_state):
        inp = input('Enter x y:')
        try:
            x, y = inp.split()
            x, y = int(x), int(y)
            return x, y
        except Exception:
            return None

import random

class RandomPlayer:
    def get_move(self, board_state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state[i][j] is None:
                    positions.append((i, j))
        return random.choice(positions)

from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2

class UserWebcamPlayer:
    def __init__(self):
        print("Initializing UserWebcamPlayer...")
        # Load model once during initialization
        self.model = models.load_model('results/Hyper_best_model.keras')
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _process_frame(self, frame):
        if frame is None:
            return None
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No face detected in frame")
            return None
            
        # Get the largest face
        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
        face_roi = frame[y:y+h, x:x+w]
        
        return face_roi

    def _access_webcam(self):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        
        if not vc.isOpened():
            print("Error: Could not access webcam")
            return None
            
        while True:
            rval, frame = vc.read()
            if not rval:
                break
                
            processed_frame = self._process_frame(frame)
            self._debug_face_detection(frame)  # Add debug visualization
            
            if processed_frame is not None:
                cv2.imshow("preview", processed_frame)
                
            key = cv2.waitKey(20)
            if key == 13:  # Enter key
                break
                
        vc.release()
        cv2.destroyAllWindows()
        return processed_frame

    def _debug_face_detection(self, frame):
        if frame is None:
            return
            
        debug_frame = frame.copy()
        gray = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add label
            cv2.putText(debug_frame, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Debug: Face Detection', debug_frame)

    def _print_reference(self, row_or_col):
        print('reference:')
        for i, emotion in enumerate(categories):
            print('{} {} is {}.'.format(row_or_col, i, emotion))
    
    def _get_row_or_col_by_text(self):
        try:
            val = int(input())
            return val
        except Exception as e:
            print('Invalid position')
            return None
    
    def _get_row_or_col(self, is_row):
        try:
            row_or_col = 'row' if is_row else 'col'
            self._print_reference(row_or_col)
            img = self._access_webcam()
            emotion = self._get_emotion(img)
            if type(emotion) is not int or emotion not in range(len(categories)):
                print('Invalid emotion number {}'.format(emotion))
                return None
            print('Emotion detected as {} ({} {}). Enter \'text\' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.'.format(categories[emotion], row_or_col, emotion))
            inp = input()
            if inp == 'text':
                return self._get_row_or_col_by_text()
            return emotion
        except Exception as e:
            # error accessing the webcam, or processing the image
            raise e
    
    def _get_emotion(self, img) -> int:
        try:
            if img is None:
                print("No valid image to process")
                return 0  # Return neutral as default
                
            # Preprocess image
            image = cv2.resize(img, (image_size[0], image_size[1]))
            
            # Convert to RGB (model expects RGB input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            image = image.astype('float32') / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            # Get prediction
            prediction = self.model.predict(image, verbose=0)
            confidence = np.max(prediction[0])
            detected_emotion = np.argmax(prediction[0])
            
            # Add confidence threshold
            if confidence < 0.6:  # Adjust threshold as needed
                print(f"Low confidence ({confidence:.2f}) - defaulting to neutral")
                return 0
                
            print(f"\nPrediction confidence: {confidence:.2f}")
            print(f"Detected emotion: {categories[detected_emotion]}")
            
            return detected_emotion
            
        except Exception as e:
            print(f"Error during emotion detection: {e}")
            return 0

    def get_move(self, board_state):
        row, col = None, None
        while row is None:
            row = self._get_row_or_col(True)
        while col is None:
            col = self._get_row_or_col(False)
        return row, col