import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
from scipy.spatial import distance

# Конфигурация
ROI_CENTER = (0.5, 0.5)  # Центр зоны интереса (относительно размера кадра)
ROI_RADIUS_RATIO = 0.3   # Радиус зоны интереса
MOVEMENT_THRESHOLD = 5.0 # Порог движения в пикселях между кадрами
FRAME_UPDATE_INTERVAL = 30 # Интервал проверки для перевыбора ID
MIN_BBOX_AREA_RATIO = 0.4 # Минимальная площадь bbox относительно начальной

# Инициализация модели
model = YOLO("hpe_models/yolo11s-pose.pt")

# Обработка видео
cap = cv2.VideoCapture("input/Heian_Nidan.mp4")
W, H = int(cap.get(3)), int(cap.get(4))

# Инициализация трекера
track_history = defaultdict(lambda: [])
selected_id = None
prev_kpts = None
frame_count = 0
initial_bbox_area = 0

# Основной цикл
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    results = model.track(frame, persist=True, verbose=False, classes=0)

    if results[0].boxes.id is None:
        continue

    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    keypoints = results[0].keypoints.xy.cpu()

    # Рассчитываем приоритеты для каждого обнаружения
    priorities = []
    for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
        # 1. Расстояние до центра ROI
        roi_center = (W*ROI_CENTER[0], H*ROI_CENTER[1])
        dist_to_center = distance.euclidean(box[:2], roi_center)

        # 2. Размер bbox
        bbox_area = box[2] * box[3]

        # 3. Движение ключевых точек
        movement = 0
        if prev_kpts is not None and selected_id == track_ids[i]:
            movement = torch.mean(np.abs(kpts - prev_kpts))

        # Собираем приоритеты (чем меньше значения - тем лучше)
        priority = [
            dist_to_center / (W*0.5),  # Нормализованное расстояние до центра
            1 - (bbox_area / (W*H)),    # Инвертированная площадь
            1 - (movement / 100)        # Инвертированное движение
        ]
        priorities.append(np.mean(priority))

    # Выбор лучшего кандидата
    if selected_id is None or frame_count % FRAME_UPDATE_INTERVAL == 0:
        if len(priorities) > 0:
            best_idx = np.argmin(priorities)
            new_id = track_ids[best_idx]
            new_area = boxes[best_idx][2] * boxes[best_idx][3]

            # Проверяем условия для переключения
            if selected_id is None or new_area > initial_bbox_area*1.2:
                selected_id = new_id
                initial_bbox_area = new_area
                bbox_threshold = initial_bbox_area * MIN_BBOX_AREA_RATIO

    # Обработка выбранного спортсмена
    if selected_id in track_ids:
        target_idx = track_ids.index(selected_id)
        target_box = boxes[target_idx]
        kpts = keypoints[target_idx]

        # Проверка условий трекинга
        bbox_area = target_box[2] * target_box[3]
        in_roi = distance.euclidean(target_box[:2], roi_center) < W*ROI_RADIUS_RATIO

        if bbox_area > bbox_threshold and in_roi:
            # Анализ движения
            if prev_kpts is not None:
                current_movement = torch.mean(np.abs(kpts - prev_kpts))
                if current_movement < MOVEMENT_THRESHOLD:
                    prev_kpts = None
                    continue

            # Визуализация
            color = (0, 255, 0)
            cv2.rectangle(frame,
                          (int(target_box[0]-target_box[2]/2), int(target_box[1]-target_box[3]/2)),
                          (int(target_box[0]+target_box[2]/2), int(target_box[1]+target_box[3]/2)),
                          color, 2)

            # Ключевые точки
            for x, y in kpts:
                cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)

            # Сохраняем ключевые точки для следующего кадра
            prev_kpts = kpts.clone()

            # Сохранение данных для анализа
            track_history[selected_id].append({
                "frame": frame_count,
                "bbox": target_box.numpy(),
                "keypoints": kpts.numpy()
            })
        else:
            # Сбрасываем при нарушении условий
            selected_id = None
            prev_kpts = None
    else:
        selected_id = None
        prev_kpts = None

    # Отрисовка ROI
    cv2.circle(frame, (int(roi_center[0]), int(roi_center[1])),
               int(W*ROI_RADIUS_RATIO), (255,0,0), 2)

    cv2.imshow("Enhanced Sport Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# Экспорт данных (пример)
with open("tracking_data.json", "w") as f:
    json.dump(track_history, f, indent=2)