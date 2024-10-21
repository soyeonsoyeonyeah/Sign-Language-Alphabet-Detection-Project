from flask import Flask
from flask import request
import base64
import json
from tensorflow import keras
import mediapipe as mp
import numpy.linalg as LA
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = keras.models.load_model('../hand_lstm_train_result')
model.summary()

seq_length = 5
gesture = {
    0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N",
    14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"
}
mp_hands = mp.solutions.hands

app = Flask(__name__)

@app.route('/lstm_detect', methods=["POST"])
def lstm_detect01():
    lstm_result = []
    seq = []

    with mp_hands.Hands() as hands:
        json_image = request.get_json()
        encoded_data_arr = json_image.get("data")
        for index, encoded_data in enumerate(encoded_data_arr):
            encoded_data = encoded_data.replace("image/jpeg;base64,", "")
            decoded_data = base64.b64decode(encoded_data)
            nparr = np.fromstring(decoded_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks != None:
                for hand_landmarks in results.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    joint1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                    joint2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                    v = joint2 - joint1
                    # 정규화 -> 1차원 배열
                    v_normal = LA.norm(v, axis=1)
                    # v와 연산을 위해 2차원 배열로 변환
                    v_normal2 = v_normal[:, np.newaxis]
                    # v를 v_normal2로 나눠 다시 정규화
                    v2 = v / v_normal2
                    a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                    b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                    ein = np.einsum('ij,ij->i', a, b)
                    # 코사인값 계산
                    radian = np.arccos(ein)
                    # 코사인값을 각도로 변환
                    angle = np.degrees(radian)
                    data = np.concatenate([joint.flatten(), angle])
                    seq.append(data)
                    if len(seq) < 5:
                        continue
                    last_seq = seq[-5:]
                    input_arr = np.array(last_seq, dtype=np.float32)
                    input_lstm_arr = input_arr.reshape(1, 5, 78)
                    y_pred = model.predict(input_lstm_arr)
                    idx = int(np.argmax(y_pred))
                    letter = gesture[idx]
                    conf = y_pred[0, idx]
                    lstm_result.append({
                        "text":f"{letter} {round(conf * 100, 2)} Percent!"
                        ,"x": int(hand_landmarks.landmark[0].x * image.shape[1])
                        ,"y": int(hand_landmarks.landmark[0].y * image.shape[0])
                    })
    return json.dumps(lstm_result)

if __name__ == '__main__':
    app.run()