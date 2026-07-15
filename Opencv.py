from inference_sdk import InferenceHTTPClient
import cv2
from dotenv import load_dotenv
import os

load_dotenv()
cap = cv2.VideoCapture(0)
CLIENT = InferenceHTTPClient(

    api_url="https://serverless.roboflow.com",
    api_key=os.getenv('ROBOFLOW_API_KEY')
)



while True:
    red, image = cap.read()
    if not red:
        print('not red')
        break

    cow = 0
    sheep = 0
    result = CLIENT.infer("image.jpg", model_id=os.getenv('MODEL_ID'))
    for i in result.get('predictions', []):
        x = i['x']
        y = i['y']
        width = i['width']
        height = i['height']
        confidence = i['confidence']
        class_name = i['class']

        # Calculate box corners (If x,y is the top-left corner)
        x1, y1 = x, y
        x2, y2 = x + width, y + height

        # Alternative: If x,y is the CENTER of the box (common in YOLO/Roboflow)
        # x1 = int(x - w / 2)
        # y1 = int(y - h / 2)
        # x2 = int(x + w / 2)
        # y2 = int(y + h / 2)

        if class_name.lower() == 'sheep':
            sheep += 1
            color = (0, 0, 255)
        elif class_name.lower() == 'cow':
            cow += 1
            color = (255, 0, 0)

        else:
            color = (0, 255, 0)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{class_name} {round(confidence, 2)*100}%', (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(image, f'sheep count: {sheep}', (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.putText(image, f'cow count: {cow}', (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)





    image = cv2.resize(image, (600, 500))
    cv2.imshow('image', image)
    if cv2.waitKey(1)& 0xFF == ord('a'):
        break



cap.release()
cv2.destroyAllWindows()