import cv2 as cv
import numpy as np
import mediapipe as mp

# 初始化MediaPipe模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 摄像头参数
capture = cv.VideoCapture(0)
focal_length = 900
actual_face_width = 0.15  # 单位：米


def select_hand(hand_results):
    """选择优先级最高的手部（优先右手）"""
    if not hand_results.multi_handedness:
        return None

    hands_info = []
    for idx, handedness in enumerate(hand_results.multi_handedness):
        hand_label = handedness.classification[0].label
        hand_score = handedness.classification[0].score
        hands_info.append({
            "label": hand_label,
            "score": hand_score,
            "landmarks": hand_results.multi_hand_landmarks[idx]
        })

    # 优先选择右手
    right_hands = [h for h in hands_info if h["label"] == "Right"]
    if right_hands:
        return max(right_hands, key=lambda x: x["score"])

    # 其次选择左手
    left_hands = [h for h in hands_info if h["label"] == "Left"]
    if left_hands:
        return max(left_hands, key=lambda x: x["score"])

    return None


def get_direction(hand_landmarks, frame_shape):
    """判断手部方向"""
    h, w = frame_shape[:2]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    cx, cy = int(wrist.x * w), int(wrist.y * h)

    finger_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]

    tip_positions = [(int(t.x * w), int(t.y * h)) for t in finger_tips]
    avg_tip = np.mean(tip_positions, axis=0)

    dx = avg_tip[0] - cx
    dy = avg_tip[1] - cy

    if abs(dx) > abs(dy):
        return "Left" if dx < 0 else "Right"
    else:
        return "Up" if dy < 0 else "Down"


def main():
    direction = "Unknown"
    direction_buffer = []
    consistent_frames = 5

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # 水平镜像翻转，使显示更符合直觉（以用户角度来判断方向）
        frame = cv.flip(frame, 1)

        # 处理手部检测
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        selected_hand = select_hand(hand_results)

        # 处理选中的手部
        if selected_hand:
            hand_landmarks = selected_hand["landmarks"]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 方向判断
            temp_direction = get_direction(hand_landmarks, frame.shape)
            direction_buffer.append(temp_direction)

            if len(direction_buffer) >= consistent_frames:
                if all(d == direction_buffer[0] for d in direction_buffer):
                    direction = direction_buffer[0]
                direction_buffer.clear()

        # 处理面部检测
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)

                # 距离估算
                if w > 0:
                    distance = (focal_length * actual_face_width) / w
                    cv.putText(frame, f"Distance: {distance:.2f}m",
                               (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示结果
        cv.putText(frame, f"Direction: {direction}", (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow("Hand & Face Detection", frame)

        if cv.waitKey(1) == 27:
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()