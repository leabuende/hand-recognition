import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
while True:
    success, img = cap.read()
    cv2.imshow("image", img)
    cv2.waitKey(1)
    if not success:
      print("No camera frame")
      continue
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            print(hand_lm)

