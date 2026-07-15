import cv2
from inference_sdk import InferenceHTTPClient

# 1. Настройка подключения к Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com/image-voewz/2",
    api_key="vbr3nNmT3yMl9TOvk0dU"  # <-- Замени на свой настоящий API_KEY
)

# 2. Запуск веб-камеры
cap = cv2.VideoCapture('../image/23_26.jpg')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Сбрасываем счетчики для каждого кадра
    cow_count = 0
    sheep_count = 0

    try:
        # Передаем frame (NumPy массив) напрямую! Библиотека сама его распознает.
        result = CLIENT.infer(frame, model_id="checkanimal-qkm62/2")
        # Разбираем результаты
        predictions = result.get("predictions", [])
        for pred in predictions:
            label = pred.get("class")

            if label == "cow":
                cow_count += 1
                color = (0, 255, 0)
            elif label == "sheep":
                sheep_count += 1
                color = (255, 191, 0)
            else:
                color = (255, 255, 255)


            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Cows: {cow_count} | Sheep: {sheep_count}", (20, 190), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                    (255, 0, 0), 1)

    except Exception as e:

        cv2.putText(frame, "Inference Error", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Ошибка распознавания: {e}")


    resized_frame = cv2.resize(frame, (700, 500))
    cv2.imshow('image', resized_frame)


    if cv2.waitKey(0) & 0xFF == ord('a'):
        break


cap.release()
cv2.destroyAllWindows()