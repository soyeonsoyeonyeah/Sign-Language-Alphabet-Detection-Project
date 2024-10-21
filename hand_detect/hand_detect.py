import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

cv2.namedWindow(winname='webcam_window01', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname='webcam_window01', width=1024, height=800)

# 화면에서 손과 손가락 관절 위치 정보 탐지하는 객체 리턴
mp_hands = mp.solutions.hands
# 인식한 손의 key point를 그릴 객체
mp_drawing = mp.solutions.drawing_utils
# 웹캠의 화면을 가져올 객체를 생성해서 리턴
cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while cap.isOpened() == True:
        success, image = cap.read()
        # 웹캠 이미지를 좌우 반전
        image = cv2.flip(image, 1)
        if success == False:
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for hand_randmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_randmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )

        cv2.imshow('webcam_window01', image)

        if cv2.waitKey(1) == ord('q'):
            break
cap.release()