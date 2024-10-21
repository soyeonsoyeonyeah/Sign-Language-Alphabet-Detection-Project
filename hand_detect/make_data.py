from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
import os
import numpy.linalg as LA
import time

# 데이터 저장 디렉토리 생성
os.makedirs('hand_dataset', exist_ok=True)
# 현재 프레임
frame = 0
# 최대 프레임 -> 600 프레임 데이터 생성 예정
MAX_FRAME = 600
# 데이터 저장 변수
all_data = []

#입력할 문자를 저장할 전역 변수
action = "미정"
# 버튼 클릭시 실행되는 함수
def btnpress():
    # 전역 변수에 값을 대입하기 위해 선언해줘야 함
    global action
    # 입력값을 전역 변수에 저장
    input = ent.get()
    action = input
    # 윈도우 창 종료
    window.destroy()

# 윈도우 창 생성
window = Tk()
#입력 박스 생성
ent = Entry(window)
# 입력 박스 추가
ent.pack()

# 라벨 생성
label = Label(window)
label.config(text="생성할 데이터의 알파벳을 입력하세요")
label.pack()

# 버튼 생성
btn = Button(window)
btn.config(text="확인")
btn.config(command=btnpress)
btn.pack()

window.mainloop()


# 손 인식 부분
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1) as hands:
    while cap.isOpened() == True:
        frame += 1
        if frame >= MAX_FRAME:
            break

        success, image = cap.read()
        image = cv2.flip(image, 1)
        if success == False:
            continue
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks != None:
            cv2.putText(
                image,
                text=f"Gathering {action} Data Frame: {MAX_FRAME - frame} Left",
                org=(0, 100),
                fontScale=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=(0,0,255),
                thickness=2
            )

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
                # 21행 3열의 0 배열 생성
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    # key point의 x,y,z좌표 저장
                    joint[j] = [lm.x, lm.y, lm.z]
                # print(joint)
                # 각 손가락 관절 간의 차 구하기
                joint1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                joint2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = joint2 - joint1
                # 정규화 -> 1차원 배열
                v_normal = LA.norm(v, axis=1)
                # v와 연산을 위해 2차원 배열로 변환
                v_normal2 = v_normal[:, np.newaxis]
                # v를 v_normal2로 나눠 다시 정규화
                v2 = v / v_normal2
                # a, b 배열의 곱을 계산
                a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                ein = np.einsum('ij,ij->i', a, b)
                # 코사인값 계산
                radian = np.arccos(ein)
                # 코사인값을 각도로 변환
                angle = np.degrees(radian)
                # 입력한 알파벳 숫자로 변환
                action_num = ord(action)
                # 알파벳을 0 ~ 25 으로 만듦
                action_label = action_num - ord('A')
                angle_label = np.append(angle, action_label)
                # 관절 좌표를 1차원 배열로 변환
                # 관절 좌표와 각도를 합쳐 data에 저장
                # 63개의 데이터는 keypoint 좌표, 15개의 데이터는 각도, 1개의 데이터는 알파벳 라벨
                data = np.concatenate([joint.flatten(), angle_label])
                all_data.append(data)


        cv2.imshow('webcam_window01', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()

# 현재날짜와 시간을 int형으로 변환
created_time = int(time.time())
# 데이터 저장
np.save(os.path.join('./hand_dataset', f'{action}-{created_time}'), all_data)