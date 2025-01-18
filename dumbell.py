import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (elbow)
    c = np.array(c)  # Last point

    # Calculate the angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))

    # Adjust if angle is over 180 degrees
    if angle > 180:
        angle = 360 - angle

    return angle

# Start capturing video from the front camera
cap = cv2.VideoCapture(1)  # Change to 1 for front camera, 0 for rear camera

round_counter = 0
correct_counter = 0
feedback_round = 3  # First 3 rounds for form correction
performance_round = 5  # Next 5 rounds for performance tracking

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the coordinates of the key points (shoulders, elbows, wrists)
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

        # Calculate the angle between shoulder, elbow, and wrist
        angle = calculate_angle(shoulder, elbow, wrist)

        # Display the angle
        cv2.putText(frame, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Feedback and counter system
        if angle < 90:  # Incomplete curl
            feedback = "Incomplete Curl! Extend your arm more!"
            correct = False
        elif 90 <= angle <= 120:  # Correct form
            feedback = "Good form!"
            correct = True
        else:  # Over-extended
            feedback = "Over-extended! Lower the weight!"
            correct = False

        # Track rounds for the first 3 rounds (form correction phase)
        if round_counter < feedback_round:
            if correct:
                correct_counter += 1  # Count correct performance
            cv2.putText(frame, f"Round {round_counter + 1}: {feedback}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            round_counter += 1  # Move to next round

        # After 3 rounds, switch to performance tracking for the next 5 rounds
        elif feedback_round <= round_counter < (feedback_round + performance_round):
            if correct:
                correct_counter += 1  # Count correct performance
            cv2.putText(frame, f"Round {round_counter + 1}: {feedback}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            round_counter += 1  # Move to next round

            # After 5 rounds, display total correct count out of 5
            if round_counter == feedback_round + performance_round:
                cv2.putText(frame, f"Correct reps: {correct_counter} out of {performance_round}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                # Reset the counter for the next rounds
                round_counter = 0
                correct_counter = 0

    # Show the frame
    cv2.imshow('Bicep Curl Performance Tracker', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
