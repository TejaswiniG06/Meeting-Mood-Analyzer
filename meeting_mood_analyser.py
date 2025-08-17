import cv2
from fer import FER
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize detector
detector = FER(mtcnn=True)

# Start webcam
cap = cv2.VideoCapture(0)

# Data storage
records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    emotions = detector.detect_emotions(frame)
    if emotions:
        top_emotion, score = detector.top_emotion(frame)

        # Save record with timestamp
        records.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Emotion": top_emotion,
            "Score": score
        })

        # Display on screen
        cv2.putText(frame, f"{top_emotion} ({score:.2f})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show webcam feed
    cv2.imshow("Meeting Mood Analyzer", frame)

    # Close when the window's close button is clicked
    if cv2.getWindowProperty("Meeting Mood Analyzer", cv2.WND_PROP_VISIBLE) < 1:
        break

    # (Optional) quit when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- Reporting ---------------- #

# Convert to DataFrame
df = pd.DataFrame(records)

if not df.empty:
    # Save detailed CSV
    df.to_csv("emotion_report.csv", index=False)

    # Save summary report
    summary = df["Emotion"].value_counts(normalize=True) * 100
    with open("emotion_summary.txt", "w") as f:
        f.write("Emotion Summary Report\n")
        f.write("=====================\n")
        for emotion, pct in summary.items():
            f.write(f"{emotion}: {pct:.2f}%\n")

    # ---------------- Timeline Graph ---------------- #
    plt.figure(figsize=(12, 6))

    # Convert Time column for plotting
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

    # One-hot encode emotions (0/1)
    emotion_dummies = pd.get_dummies(df["Emotion"])
    timeline = pd.concat([df["Time"], emotion_dummies], axis=1)

    # Plot each emotion separately
    for emotion in emotion_dummies.columns:
        plt.plot(timeline["Time"], timeline[emotion],
                 marker="o", linestyle="-", label=emotion)

    plt.title("Emotion Timeline During Meeting")
    plt.xlabel("Time")
    plt.ylabel("Detected Emotion (1=Present, 0=Absent)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("emotion_timeline.png")
    plt.show()

    print("\n✅ Reports generated:")
    print("- emotion_report.csv (detailed log)")
    print("- emotion_summary.txt (summary stats)")
    print("- emotion_timeline.png (graph)")
else:
    print("⚠️ No emotions detected during session.")
