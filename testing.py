import cv2
from fer import FER
from collections import Counter

# Initialize detector and webcam
detector = FER()
cap = cv2.VideoCapture(0)

# Emotion counter
emotion_counter = Counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)
    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        # Update counter
        emotion_counter[top_emotion] += 1

        # Draw face box + emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, top_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show stats in top-left corner
    y0 = 30
    for i, (emo, count) in enumerate(emotion_counter.most_common(5)):
        text = f"{emo}: {count}"
        cv2.putText(frame, text, (10, y0 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Emotion Recognition", frame)

    # Exit when pressing 'q' OR when closing the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Emotion Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
