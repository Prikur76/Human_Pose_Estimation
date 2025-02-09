import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class PoseTracker:
    def __init__(self, config):
        # Инициализация моделей
        self.detector = YOLO(config['detector_model'])
        self.tracker = DeepSort(**config['tracker_params'])

        # Конфигурация
        self.target_count = config['target_count']
        self.min_keypoints_conf = config['min_keypoints_conf']
        self.target_tracks = deque(maxlen=self.target_count)

        # Фильтры и буферы
        self.track_history = {}
        self.colors = [(0,255,0), (0,0,255), (255,0,0)]

    def process_frame(self, frame):
        # Этап 1: Детекция объектов
        detections = self._detect_objects(frame)

        # Этап 2: Трекинг объектов
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Этап 3: Выбор целевых треков
        self._select_targets(tracks, frame.shape)

        # Этап 4: Анализ движений
        analysis = self._analyze_movements(frame)

        # Этап 5: Визуализация
        return self._visualize(frame, analysis)

    def _detect_objects(self, frame):
        results = self.detector.track(frame, persist=True, verbose=False, classes=0)
        detections = []

        if results[0].boxes.id is not None:
            for box, kpts, track_id in zip(results[0].boxes.xywh,
                                           results[0].keypoints.xy,
                                           results[0].boxes.id.int().cpu().tolist()):
                if np.mean(kpts[:, 2]) < self.min_keypoints_conf:
                    continue

                detections.append({
                    'bbox': box.cpu().numpy(),
                    'keypoints': kpts.cpu().numpy(),
                    'track_id': track_id
                })
        return detections

    def _select_targets(self, tracks, frame_shape):
        current_tracks = []
        h, w = frame_shape[:2]

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()
            score = self._calculate_track_score(bbox, track, (w, h))
            current_tracks.append((score, track))

        current_tracks.sort(reverse=True, key=lambda x: x[0])
        self.target_tracks = [ct[1] for ct in current_tracks[:self.target_count]]

    def _calculate_track_score(self, bbox, track, frame_size):
        w, h = frame_size
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        center = ((bbox[0]+bbox[2])/2/w, (bbox[1]+bbox[3])/2/h)

        return (
                area * 0.5 +                   # Приоритет крупным объектам
                (1 - abs(center[0]-0.5)) * 0.3 + # Центральное положение
                len(track.hits) * 0.2           # Стабильность трека
        )

    def _analyze_movements(self, frame):
        analysis = {}
        for i, track in enumerate(self.target_tracks):
            if not hasattr(track, 'keypoints'):
                continue

            # Анализ шага
            analysis[f'track_{i}'] = {
                'step': self._detect_step(track.keypoints),
                'angles': self._calculate_joint_angles(track.keypoints)
            }
        return analysis

    def _detect_step(self, keypoints):
        # Логика определения шага (пример)
        hip = keypoints[11]  # Левый тазобедренный сустав
        ankle = keypoints[15] # Левая лодыжка
        return ankle[1] - hip[1] > 0.1  # Упрощенная логика

    def _calculate_joint_angles(self, keypoints):
        # Расчет углов основных суставов
        angles = {}

        # Пример: угол левого колена
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        angles['left_knee'] = self._vector_angle(hip, knee, ankle)

        return angles

    def _vector_angle(self, a, b, c):
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
        return np.degrees(np.arccos(cosine))

    def _visualize(self, frame, analysis):
        # Отрисовка треков
        for i, track in enumerate(self.target_tracks):
            color = self.colors[i % len(self.colors)]
            bbox = track.to_ltrb()

            # Рамка
            cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)

            # Ключевые точки
            if hasattr(track, 'keypoints'):
                for kp in track.keypoints:
                    x, y = kp[:2]
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)

            # Информация об анализе
            if f'track_{i}' in analysis:
                info = analysis[f'track_{i}']
                text = f"Step: {info['step']} | Angles: {info['angles']}"
                cv2.putText(frame, text,
                            (int(bbox[0]), int(bbox[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

# Конфигурация
config = {
    'detector_model': 'yolov8n-pose.pt',
    'tracker_params': {
        'max_age': 30,
        'n_init': 3,
        'max_cosine_distance': 0.4,
        'embedder': 'mobilenet'
    },
    'target_count': 2,
    'min_keypoints_conf': 0.3
}

# Пример использования
if __name__ == "__main__":
    tracker = PoseTracker(config)
    cap = cv2.VideoCapture("sport.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = tracker.process_frame(frame)
        cv2.imshow("Pose Tracking", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()