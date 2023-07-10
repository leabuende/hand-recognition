import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    success, img = cap.read()
    if not success:
      print("No camera frame")
      continue
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            for id, lm in enumerate(hand_lm.landmark):
                h, w, c = image.shape
                cx, cy = int(w*lm.x), int(h*lm.y)
                if id == 8 :
                    #print(id, cx, cy, lm.z)
                    if (lm.z < -0.2):
                        print('tap')
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(id), (cx, cy), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    cv2.waitKey(1)
