import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Конфигурация
ROI_CENTER = (0.5, 0.5)       # Центр зоны интереса
ROI_RADIUS_RATIO = 0.3        # Радиус зоны интереса
MIN_TRACK_LENGTH = 15         # Минимальная длина трека для стабилизации
MOVEMENT_THRESHOLD = 0.02     # Порог движения (нормализованный)

# Инициализация MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Для множественных детекций:
# mp_holistic = mp.solutions.holistic
# with mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=2,
#     enable_segmentation=True,
#     refine_face_landmarks=True) as holistic:

class PoseTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = 5

    def update(self, detections):
        # Простейший трекинг по расстоянию между центрами
        updated_tracks = {}

        for detection in detections:
            min_dist = float('inf')
            best_id = None

            for track_id, track in self.tracks.items():
                last_pos = track['positions'][-1]
                dist = np.linalg.norm(detection['center'] - last_pos)

                if dist < min_dist and dist < 0.1:  # Порог расстояния
                    min_dist = dist
                    best_id = track_id

            if best_id is not None:
                track = self.tracks[best_id]
                track['positions'].append(detection['center'])
                track['disappeared'] = 0
                updated_tracks[best_id] = track
                del self.tracks[best_id]
            else:
                new_id = self.next_id
                updated_tracks[new_id] = {
                    'positions': deque([detection['center']], maxlen=30),
                    'bbox': detection['bbox'],
                    'keypoints': detection['keypoints'],
                    'disappeared': 0
                }
                self.next_id += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] < self.max_disappeared:
                self.tracks[track_id]['disappeared'] += 1
                updated_tracks[track_id] = self.tracks[track_id]

        self.tracks = updated_tracks
        return self.tracks

# Основной код обработки видео
cap = cv2.VideoCapture("input/Heian_Nidan.mp4")
W, H = int(cap.get(3)), int(cap.get(4))

tracker = PoseTracker()
selected_track = None

with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1,
        enable_segmentation=False,
        static_image_mode=False) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Для обработки видео высокой четкости:
        # Ресайз кадра перед обработкой
        # resized_frame = cv2.resize(frame, (640, 360))
        # Восстановление координат после обработки...

        # Конвертация цвета и обработка
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        detections = []
        if results.pose_landmarks:
            # Для множественных детекций нужно использовать решение Holistic
            landmarks = results.pose_landmarks.landmark

            # Рассчет bbox на основе ключевых точек
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            center = np.array([(min_x + max_x)/2, (min_y + max_y)/2])
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

            detections.append({
                'center': center,
                'bbox': bbox,
                'keypoints': [(lm.x, lm.y, lm.visibility) for lm in landmarks]
            })

        # Обновление трекера
        tracks = tracker.update(detections)

        # Выбор лучшего трека
        best_score = -1
        current_best = None

        for track_id, track in tracks.items():
            # Рассчет критериев
            positions = np.array(track['positions'])

            # 1. Длина трека
            track_length = len(positions)

            # 2. Центральное положение
            roi_center = np.array(ROI_CENTER)
            dist_to_center = np.linalg.norm(positions[-1] - roi_center)

            # 3. Размер bbox
            bbox_size = track['bbox'][2] * track['bbox'][3]

            # 4. Движение
            if track_length > 1:
                movement = np.mean(np.abs(positions[-1] - positions[-2]))
            else:
                movement = 0

            # Итоговый score
            score = (track_length * 0.4 +
                     (1 - dist_to_center) * 0.3 +
                     bbox_size * 0.2 +
                     movement * 0.1)

            if score > best_score:
                best_score = score
                current_best = track_id

        # Обновление выбранного трека
        if current_best is not None and best_score > 0.5:
            selected_track = current_best
        else:
            selected_track = None

        # Визуализация
        if selected_track is not None and selected_track in tracks:
            track = tracks[selected_track]
            bbox = track['bbox']

            # Конвертация нормализованных координат
            x_min = int(bbox[0] * W)
            y_min = int(bbox[1] * H)
            width = int(bbox[2] * W)
            height = int(bbox[3] * H)

            # Отрисовка bbox
            cv2.rectangle(frame,
                          (x_min, y_min),
                          (x_min + width, y_min + height),
                          (0, 255, 0), 2)

            # Отрисовка ключевых точек
            for kp in track['keypoints']:
                x = int(kp[0] * W)
                y = int(kp[1] * H)
                if kp[2] > 0.5:  # Порог видимости
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Отрисовка трека
            for pos in track['positions']:
                x = int(pos[0] * W)
                y = int(pos[1] * H)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Отображение ROI
        roi_center = (int(W * ROI_CENTER[0]), int(H * ROI_CENTER[1]))
        roi_radius = int(W * ROI_RADIUS_RATIO)
        cv2.circle(frame, roi_center, roi_radius, (255, 0, 0), 2)

        cv2.imshow('MediaPipe Sport Tracking', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()