import glob
import numpy as np
import cv2
import mediapipe as mp
import numpy.linalg as LA

path = "./hand_dataset/*.npy"
file_list = glob.glob(path)

all_data = []
for file_path in file_list:
    # print(file_path)
    # file_path에 저장된 파일의 내용 읽어서 data에 저장
    data = np.load(file_path)
    # 데이터 추가
    all_data.extend(data)

save_data = np.array(all_data, dtype=np.float32)
angle_arr = save_data[:, 63:-1]
label_arr = save_data[:, -1]

gesture = {
    0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N",
    14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"
}

# KNN 객체 생성
knn = cv2.ml.KNearest_create()
knn.train(angle_arr, cv2.ml.ROW_SAMPLE, label_arr)


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
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
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
                data = np.array([angle], dtype=np.float32)
                # 현재 손의 각도 data와 KNN에 저장된 angle_arr과 거리를 계산하고 가장 가까운 3개 조회
                # 가장 가까운 손모양 결과(실수값), 가장 가까운 손모양 결과(배열), 거리가 가장 가까운 3개 손모양 결과, 가장 가까운 거리 3개
                retval, results, neighbours, dis = knn.findNearest(data, 3)

                idx = int(retval)
                if 0 <= idx <= 9:
                    cv2.putText(
                        image,
                        text=gesture[idx],
                        org=(
                            int(hand_landmarks.landmark[0].x * image.shape[1]),
                            int(hand_landmarks.landmark[0].y * image.shape[0])
                        ),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0,0,255),
                        thickness=2
                    )

        cv2.imshow('webcam_window01', image)

        if cv2.waitKey(1) == ord('q'):
            break
cap.release()