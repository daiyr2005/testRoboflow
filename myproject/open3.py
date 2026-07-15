import cv2
from inference_sdk import InferenceHTTPClient

# 1. Настройка подключения к Roboflow (Исправлено!)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # Оставляем только базовый URL!
    api_key="vbr3nNmT3yMl9TOvk0dU"          # Твой API-ключ
)

# 2. Источник видео (укажи '0' для веб-камеры или путь к файлу/картинке)
cap = cv2.VideoCapture('../image/23_26.jpg')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось загрузить кадр / видео завершено")
        break

    # Словари для подсчета объектов на текущем кадре
    traffic_light_stats = {"red": 0, "yellow": 0, "green": 0, "off": 0}
    stop_sign_count = 0

    try:
        # Используем готовую качественную модель для светофоров и знаков
        # Классы этой модели: 'Green Light', 'Red Light', 'Yellow Light', 'Stop Sign' и др.
        result = CLIENT.infer(frame, model_id="traffic-lights-vttga/1")
        predictions = result.get("predictions", [])

        for pred in predictions:
            raw_label = pred.get("class", "").lower()
            confidence = pred.get("confidence", 0)

            # Фильтруем слишком неуверенные предсказания (порог 40%)
            if confidence < 0.4:
                continue

            # Определяем тип объекта и цвет рамки
            label = "Object"
            color = (255, 255, 255) # Белый по умолчанию

            if "red" in raw_label:
                label = "Red Light"
                traffic_light_stats["red"] += 1
                color = (0, 0, 255) # Красный (BGR)
            elif "yellow" in raw_label or "amber" in raw_label:
                label = "Yellow Light"
                traffic_light_stats["yellow"] += 1
                color = (0, 255, 255) # Желтый (BGR)
            elif "green" in raw_label:
                label = "Green Light"
                traffic_light_stats["green"] += 1
                color = (0, 255, 0) # Зеленый (BGR)
            elif "stop" in raw_label:
                label = "STOP SIGN"
                stop_sign_count += 1
                color = (0, 0, 139) # Темно-красный для знака STOP

            # Вычисление координат Bounding Box
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            # Отрисовка рамки и текста
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Вывод статистики на экран
        stats_text = (
            f"STOP: {stop_sign_count} | "
            f"RED: {traffic_light_stats['red']} | "
            f"YELLOW: {traffic_light_stats['yellow']} | "
            f"GREEN: {traffic_light_stats['green']}"
        )
        cv2.putText(frame, stats_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception as e:
        cv2.putText(frame, "Inference Error", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"Ошибка распознавания: {e}")

    # Изменяем размер для вывода на экран
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow('Traffic Control Detection', resized_frame)

    # Нажмите 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()