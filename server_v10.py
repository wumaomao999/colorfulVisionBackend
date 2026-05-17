#v1.0
import os

import cv2
import mediapipe as mp
from flask import Flask, jsonify, request
from flask_cors import CORS

# 创建 Flask 应用并启用 CORS
app = Flask(__name__)
CORS(app)  # 允许所有域名的请求

# 设置上传文件的存储目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 根路由
@app.route('/')
def home():
    return "你好，这里是睛彩测视app后端公网IP。————xiewu"


# 上传文件接口
@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "failure", "message": "没有文件上传"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"status": "failure", "message": "没有选择文件"}), 400

        # 保存文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        hand = request.headers.get("hand")

        direction, face_distance = analyze_image(file_path, hand)
        print("direction:", direction, "face distance:", face_distance)

        os.remove(file_path)
        return jsonify(
            {"status": "success", "message": "文件上传成功", "direction": direction, "face_distance": face_distance})
    except Exception as e:
        print(f"处理上传请求时发生错误: {e}")
        return jsonify({"status": "failure", "message": "服务器错误"}), 500


def analyze_image(path: str, hand: int) -> (str, float):
    """
    手部和面部检测封装方法
    :param path: 图片路径
    :param hand: 0-优先右手，1-优先左手
    :return: (方向, 面部距离)
    """
    # 初始化模型
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
    ) as hand_model, mp_face.FaceDetection(
        min_detection_confidence=0.5
    ) as face_model:

        # 读取并处理图像
        img = cv2.imread(path)
        if img is None:
            return "Unknown", 0.0

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 手部检测
        hand_results = hand_model.process(rgb_img)
        selected_hand = None

        if hand_results.multi_hand_landmarks:
            hands = []
            for idx, handedness in enumerate(hand_results.multi_handedness):
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                hands.append({
                    "label": label,
                    "score": score,
                    "landmarks": hand_results.multi_hand_landmarks[idx]
                })

            # 根据参数选择手
            target_label = "Right" if hand == 0 else "Left"
            candidates = [h for h in hands if h["label"] == target_label]
            if candidates:
                selected_hand = max(candidates, key=lambda x: x["score"])
            else:
                if hands:
                    selected_hand = max(hands, key=lambda x: x["score"])

        # 方向判断
        direction = "Unknown"
        if selected_hand:
            landmarks = selected_hand["landmarks"]
            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            dx = index_tip.x - wrist.x
            dy = index_tip.y - wrist.y

            if abs(dx) > abs(dy):
                direction = "Left" if dx < 0 else "Right"
            else:
                direction = "Up" if dy < 0 else "Down"

        # 面部距离计算
        face_distance = 0.0
        face_results = face_model.process(rgb_img)
        if face_results.detections:
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            face_width = bbox.width * w
            if face_width > 0:
                face_distance = round((800 * 0.15) / face_width, 2)  # 焦距900px，实际面部宽度0.15m

        # 控制台输出
        print(f"Direction: {direction}")
        print(f"Face Distance: {face_distance}m")

        return direction, face_distance


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8090)
