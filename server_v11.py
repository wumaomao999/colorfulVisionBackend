#v1.1
import io
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, request
from flask_cors import CORS

# 创建 Flask 应用并启用 CORS
app = Flask(__name__)
CORS(app)  # 允许所有域名的请求


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

        # 将文件内容转换为内存字节流
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)  # 重置指针到文件开头

        hand = request.headers.get("hand", type=int)

        # 将字节流转换为OpenCV图像
        img_array = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        direction, face_distance = analyze_image(img, hand)
        print("direction:", direction, "face distance:", face_distance)

        return jsonify(
            {"status": "success", "message": "文件上传成功",
             "direction": direction, "face_distance": face_distance})

    except Exception as e:
        print(f"处理上传请求时发生错误: {e}")
        return jsonify({"status": "failure", "message": "服务器错误"}), 500


def analyze_image(img: np.ndarray, hand: int) -> (str, float):
    """
    手部和面部检测封装方法（内存处理版）
    :param img: OpenCV图像矩阵
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

        if img is None:
            return "Unknown", 0.0

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 手部检测（与原始逻辑保持一致）
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

            target_label = "Right" if hand == 0 else "Left"
            candidates = [h for h in hands if h["label"] == target_label]
            selected_hand = max(candidates, key=lambda x: x["score"]) if candidates else max(hands, key=lambda x: x[
                "score"]) if hands else None

        # 方向判断（与原始逻辑保持一致）
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

        # 面部距离计算（与原始逻辑保持一致）
        face_distance = 0.0
        face_results = face_model.process(rgb_img)
        if face_results.detections:
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            face_width = bbox.width * w
            if face_width > 0:
                face_distance = round((800 * 0.15) / face_width, 2)

        return direction, face_distance


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8090)