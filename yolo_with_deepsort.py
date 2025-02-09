import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Доступные варианты:
# - "mobilenet" (быстрая)
# - "torchreid" (точная)
# - путь к кастомной ONNX модели
# embedder="mobilenet"

class SportsTracker:
    def __init__(self):
        # Инициализация моделей
        self.detector = YOLO('hpe_models/yolo11n-pose.pt')
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )

        # Конфигурация
        self.ROI = (0.25, 0.25, 0.75, 0.75)  # x1, y1, x2, y2
        self.MIN_KEYPOINTS_CONF = 0.3
        self.TARGET_TRACK_ID = None

    def process_frame(self, frame):
        # Детекция объектов
        detections = self.detect_objects(frame)

        # Трекинг с DeepSORT
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Выбор целевого трека
        self.select_main_track(tracks, frame.shape)

        # Отрисовка результатов
        return self.draw_results(frame, tracks)

    def detect_objects(self, frame):
        # Получение детекций от YOLO
        results = self.detector.track(frame, persist=True, verbose=False, classes=0)

        detections = []
        if results[0].boxes.id is not None:
            for box, kpts, track_id in zip(results[0].boxes.xywh,
                                           results[0].keypoints.xy,
                                           results[0].boxes.id.int().cpu().tolist()):
                x, y, w, h = box.cpu().numpy()
                conf = box.conf.cpu().numpy()[0]

                # Фильтрация по ключевым точкам
                if np.mean(kpts[:, 2]) < self.MIN_KEYPOINTS_CONF:
                    continue

                detections.append({
                    'bbox': (x - w/2, y - h/2, w, h),
                    'confidence': conf,
                    'keypoints': kpts.cpu().numpy(),
                    'class_id': 0,
                    'track_id': track_id
                })
        return detections

    def select_main_track(self, tracks, frame_shape):
        current_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            # Проверка нахождения в ROI
            bbox = track.to_ltrb()
            roi_x1, roi_y1, roi_x2, roi_y2 = self.get_absolute_roi(frame_shape)
            in_roi = (bbox[0] > roi_x1) and (bbox[1] > roi_y1) and \
                     (bbox[2] < roi_x2) and (bbox[3] < roi_y2)

            if not in_roi:
                continue

            # Расчет приоритета
            track_length = len(track.hits)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            current_tracks.append({
                'track': track,
                'score': track_length + bbox_area * 0.001
            })

        if current_tracks:
            best_track = max(current_tracks, key=lambda x: x['score'])
            self.TARGET_TRACK_ID = best_track['track'].track_id

    def get_absolute_roi(self, frame_shape):
        h, w = frame_shape[:2]
        return (
            int(w * self.ROI[0]),
            int(h * self.ROI[1]),
            int(w * self.ROI[2]),
            int(h * self.ROI[3])
        )

    def draw_results(self, frame, tracks):
        # Отрисовка ROI
        roi_coords = self.get_absolute_roi(frame.shape)
        cv2.rectangle(frame,
                      (roi_coords[0], roi_coords[1]),
                      (roi_coords[2], roi_coords[3]),
                      (255, 0, 0), 2)

        # Отрисовка треков
        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()
            track_id = track.track_id
            color = (0, 255, 0) if track_id == self.TARGET_TRACK_ID else (0, 0, 255)

            # Отрисовка bbox
            cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)

            # Отрисовка ключевых точек
            if track_id == self.TARGET_TRACK_ID and hasattr(track, 'keypoints'):
                for kp in track.keypoints:
                    x, y, conf = kp
                    if conf > self.MIN_KEYPOINTS_CONF:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        return frame

if __name__ == "__main__":
    tracker = SportsTracker()
    cap = cv2.VideoCapture("input/Heian_Nidan.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = tracker.process_frame(frame)
        cv2.imshow("Sports Tracking", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()