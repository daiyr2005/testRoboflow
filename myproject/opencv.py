import cv2
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
load_dotenv()

# 1. Initialize the client
CLIENT = InferenceHTTPClient(
    api_url=os.getenv('ROBOFLOW_URL'),
    api_key=os.getenv('ROBOFLOW_API_KEY')  # Replace with your real key
)

# 2. Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 3. Pass the raw numpy frame directly to the inference client
    # Roboflow's SDK handles the conversion behind the scenes!
    try:
        result = CLIENT.infer(frame, model_id="model-id/1")

        # Initialize counts for this specific frame
        cow_count = 0
        sheep_count = 0

        # 4. Parse the predictions (Assuming standard Roboflow object detection JSON structure)
        predictions = result.get("predictions", [])
        for pred in predictions:
            label = pred.get("class")

            # Count based on label name (Adjust string names to match your exact classes)
            if label == "cow":
                cow_count += 1
            elif label == "sheep":
                sheep_count += 1

            # Optional: Draw bounding boxes on the frame
            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

            # Convert center x,y to top-left corner coordinates for OpenCV
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw the box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 5. Display counts overlay on video
        cv2.putText(frame, f"Cows: {cow_count} | Sheep: {sheep_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    except Exception as e:
        cv2.putText(frame, "Inference Error", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Error: {e}")

    # Show the frame
    cv2.imshow('Livestock Counter', frame)

    # Break the loop if 'a' is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()