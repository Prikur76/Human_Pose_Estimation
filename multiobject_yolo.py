# Реализация трекинга определенного количества объектов (N) с настройками

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from collections import deque


class MultiObjectTracker:
    def __init__(self, target_count=2):
        # Инициализация моделей
        self.detector = YOLO('yolov8n-pose.pt')
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4,
            embedder="mobilenet"
        )

        # Конфигурация
        self.TARGET_COUNT = target_count  # Желаемое количество отслеживаемых объектов
        self.MIN_KEYPOINTS_CONF = 0.3
        self.target_tracks = deque(maxlen=self.TARGET_COUNT)
        self.colors = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (0,255,255)]

    def process_frame(self, frame):
        # Детекция и трекинг
        detections = self._prepare_detections(frame)
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Выбор целевых треков
        self._update_target_tracks(tracks, frame.shape)

        # Визуализация
        return self._draw_results(frame)

    def _prepare_detections(self, frame):
        results = self.detector.track(frame, persist=True, verbose=False, classes=0)
        detections = []

        if results[0].boxes.id is not None:
            for box, kpts, track_id in zip(results[0].boxes.xywh,
                                           results[0].keypoints.xy,
                                           results[0].boxes.id.int().cpu().tolist()):
                if np.mean(kpts[:, 2]) < self.MIN_KEYPOINTS_CONF:
                    continue

                detections.append({
                    'bbox': box.cpu().numpy(),
                    'confidence': box.conf.cpu().numpy()[0],
                    'keypoints': kpts.cpu().numpy(),
                    'class_id': 0,
                    'track_id': track_id
                })
        return detections

    def _update_target_tracks(self, tracks, frame_shape):
        current_tracks = []
        h, w = frame_shape[:2]

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            center = ((bbox[0]+bbox[2])/2/w, (bbox[1]+bbox[3])/2/h)

            # Критерии оценки
            score = (
                    area * 0.5 +                   # Размер объекта
                    (1 - abs(center[0]-0.5)) * 0.3 + # Близость к центру
                    len(track.hits) * 0.2           # Стабильность трека
            )

            current_tracks.append((score, track))

        # Сортировка по убыванию score
        current_tracks.sort(reverse=True, key=lambda x: x[0])

        # Выбор топ-N треков
        self.target_tracks.clear()
        for i in range(min(self.TARGET_COUNT, len(current_tracks))):
            self.target_tracks.append(current_tracks[i][1])

    def _draw_results(self, frame):
        # Отрисовка всех целевых треков
        for i, track in enumerate(self.target_tracks):
            color = self.colors[i % len(self.colors)]
            bbox = track.to_ltrb()

            # Рамка
            cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)

            # ID и номер объекта
            cv2.putText(frame, f"ID:{track.track_id} OBJ:{i+1}",
                        (int(bbox[0]), int(bbox[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Ключевые точки
            if hasattr(track, 'keypoints'):
                for kp in track.keypoints:
                    x, y, conf = kp
                    if conf > self.MIN_KEYPOINTS_CONF:
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        return frame

# Использование
tracker = MultiObjectTracker(target_count=2)  # Задаем количество отслеживаемых объектов

cap = cv2.VideoCapture("sport.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = tracker.process_frame(frame)
    cv2.imshow("Multi Object Tracking", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Дополнительные возможности:
# Динамическое изменение количества:
# # В процессе обработки можно менять target_count
# tracker.TARGET_COUNT = 3
# Фильтрация по размеру:
# MIN_AREA = 0.1 * (frame_width * frame_height)  # 10% площади кадра
# if area < MIN_AREA:
#     continue
# Сохранение данных:
# def save_tracks_data(self, filename):
#     data = {
#         str(track.track_id): {
#             "bboxes": [track.to_ltrb()],
#             "keypoints": track.keypoints if hasattr(track, 'keypoints') else []
#         } for track in self.target_tracks
#     }
#     import json
#     with open(filename, 'w') as f:
#         json.dump(data, f)

# Советы по настройке:
# Для улучшения стабильности:
# self.tracker = DeepSort(
#     max_age=15,  # Уменьшите для быстрого забывания потерянных объектов
#     n_init=5     # Увеличьте для более строгой инициализации треков
# )
# Для работы с пересекающимися объектами:
# # В методе _update_target_tracks добавьте проверку на overlap
# def _is_overlapping(self, bbox1, bbox2):
#     x1_min, y1_min, x1_max, y1_max = bbox1
#     x2_min, y2_min, x2_max, y2_max = bbox2
#     return not (x1_max < x2_min or x1_min > x2_max or
#                 y1_max < y2_min or y1_min > y2_max)
# Для спортивных применений добавьте фильтрацию по типу движения:
# Анализ ключевых точек для определения типа активности
# def _is_running(self, keypoints):
#     # Логика определения бега по положению ног
#     pass