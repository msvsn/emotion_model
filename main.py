import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

@dataclass
class FaceCoordinates:
    x: int
    y: int
    width: int
    height: int

class EmotionDetector:
    def __init__(self):
        self._emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        model_path = 'emotion_model.keras'
        self.model = load_model(model_path)
        
    def preprocess_image(self, face_image):
        # Перетворюємо зображення для нейромережі
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.expand_dims(face_image, axis=-1)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = face_image / 255.0
        return face_image
        
    def detect_emotion(self, face_image) -> str:
        processed_image = self.preprocess_image(face_image)
        prediction = self.model.predict(processed_image)
        emotion_index = np.argmax(prediction[0])
        emotion = self._emotions[emotion_index]
        
        # Мапінг англійських емоцій на українські
        emotion_mapping = {
            'angry': 'злий',
            'disgust': 'злий',  # Об'єднуємо з "злий"
            'fear': 'сумний',   # Об'єднуємо з "сумний"
            'happy': 'щасливий',
            'neutral': 'нейтральний',
            'sad': 'сумний',
            'surprise': 'щасливий'  # Об'єднуємо з "щасливий"
        }
        
        return emotion_mapping[emotion]

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
    def detect_face(self, image) -> Optional[FaceCoordinates]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = FaceCoordinates(
                x=int(bboxC.xmin * iw),
                y=int(bboxC.ymin * ih),
                width=int(bboxC.width * iw),
                height=int(bboxC.height * ih)
            )
            return bbox
        return None

class SarcasticResponder:
    def __init__(self):
        self._responses = {
            'щасливий': [
                "О, як мило! Але всі ми знаємо, що це маска. Твій рік був ще гіршим, ніж у Санти без сну.",
                "Ваша усмішка така фальшива, що навіть Санта не обдуриться."
            ],
            'сумний': [
                "Сумно, але чесно. Вітаємо, Крампус вже в дорозі!",
                "Нарешті хтось чесний! Але це не допоможе."
            ],
            'злий': [
                "Різдво? Ти вже у списку тих, кому подарують лише вугілля.",
                "Твій рівень злості перевищує температуру спалення печива!"
            ]
        }
    
    def get_response(self, emotion: str) -> str:
        return random.choice(self._responses.get(emotion, ["Щось пішло не так..."]))

class ChristmasFilter:
    def __init__(self):
        self.filters = {
            'santa_hat': cv2.imread('images/santa_hat.png', -1),
            'horns': cv2.imread('images/horns.png', -1),
            'snowman': cv2.imread('images/snowman.png', -1)
        }
        self.current_filter = 'santa_hat'
        
    def overlay_image(self, background, overlay, x, y, scale=1.0):
        if overlay is None:
            return background
            
        # Масштабуємо фільтр
        h, w = overlay.shape[:2]
        overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
        h, w = overlay.shape[:2]
        
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return background
            
        # Накладаємо фільтр з урахуванням альфа-каналу
        if overlay.shape[2] == 4:  # З альфа-каналом
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * \
                    (1 - alpha) + overlay[:, :, c] * alpha
        return background
        
    def apply_filter(self, image, face_coords: FaceCoordinates):
        if self.filters[self.current_filter] is None:
            return image
            
        # Розраховуємо позицію фільтра відносно обличчя
        filter_width = int(face_coords.width * 1.5)
        scale = filter_width / self.filters[self.current_filter].shape[1]
        
        if self.current_filter == 'santa_hat':
            x = face_coords.x - int(filter_width * 0.25)
            y = face_coords.y - int(filter_width * 0.7)
        elif self.current_filter == 'horns':
            x = face_coords.x - int(filter_width * 0.25)
            y = face_coords.y - int(filter_width * 0.5)
        else:  # snowman
            x = face_coords.x
            y = face_coords.y - int(filter_width * 0.2)
            
        return self.overlay_image(image, self.filters[self.current_filter], x, y, scale)
        
    def switch_filter(self):
        filters = list(self.filters.keys())
        current_index = filters.index(self.current_filter)
        self.current_filter = filters[(current_index + 1) % len(filters)]

class KarmaSystem:
    def __init__(self):
        self.karma_score = 50  # Початковий рівень карми
        self.toxicity = 0
        self.generosity = 0
        self.passive_aggression = 0
        
    def update_karma(self, emotion: str):
        # Оновлюємо показники на основі емоцій
        if emotion == 'щасливий':
            self.karma_score += 2
            self.generosity += 1
        elif emotion == 'злий':
            self.karma_score -= 3
            self.toxicity += 2
        elif emotion == 'сумний':
            self.karma_score -= 1
            self.passive_aggression += 1
            
        # Обмежуємо значення
        self.karma_score = max(0, min(100, self.karma_score))
        self.toxicity = max(0, min(100, self.toxicity))
        self.generosity = max(0, min(100, self.generosity))
        self.passive_aggression = max(0, min(100, self.passive_aggression))
        
    def get_karma_status(self) -> dict:
        return {
            'karma': self.karma_score,
            'toxicity': self.toxicity,
            'generosity': self.generosity,
            'passive_aggression': self.passive_aggression
        }

class SantaJudgementDay:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.emotion_detector = EmotionDetector()
        self.responder = SarcasticResponder()
        self.filter = ChristmasFilter()
        self.karma_system = KarmaSystem()
        self.cap = cv2.VideoCapture(0)
        
    def draw_karma_stats(self, frame, stats: dict):
        # Відображаємо статистику карми
        y_offset = 50
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            face_coords = self.face_detector.detect_face(frame)
            if face_coords:
                frame = self.filter.apply_filter(frame, face_coords)
                face_image = frame[face_coords.y:face_coords.y + face_coords.height,
                                 face_coords.x:face_coords.x + face_coords.width]
                emotion = self.emotion_detector.detect_emotion(face_image)
                response = self.responder.get_response(emotion)
                
                # Оновлюємо карму
                self.karma_system.update_karma(emotion)
                karma_stats = self.karma_system.get_karma_status()
                
                # Відображаємо все на екрані
                cv2.putText(frame, response, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.draw_karma_stats(frame, karma_stats)
            
            cv2.imshow('Santa Judgement Day', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.filter.switch_filter()
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SantaJudgementDay()
    app.run()
