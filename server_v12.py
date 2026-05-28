import io
import os
import secrets
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, abort, g, jsonify, request
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "vision_app.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"
GUIDE_PATH = BASE_DIR / "resource" / "guide.pdf"
TOKEN_TTL_DAYS = 30


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def get_mediapipe_solutions():
    # 某些 Python 环境虽然能导入 mediapipe，但不会暴露经典的
    # Solutions API。这里尽早报错，并直接给出正确启动方式。
    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        raise RuntimeError(
            "MediaPipe Solutions API is unavailable in the current Python environment. "
            "Start this backend with the project virtual environment: .venv\\Scripts\\python houduan\\server_v12.py"
        )
    return solutions


class VisionAnalyzer:
    def __init__(self):
        solutions = get_mediapipe_solutions()
        self._hands_module = solutions.hands
        # 使用 Face Mesh 替代 Face Detection
        self._face_mesh_module = solutions.face_mesh
        # 分析器在所有请求之间复用，避免每次请求都重复初始化模型。
        self._hands = self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        # 初始化 Face Mesh，启用平滑 landmark 以提高稳定性
        self._face_mesh = self._face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # 关键：提供更精确的眼睑轮廓点
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # MediaPipe 对象不保证线程安全，因此推理过程统一加锁。
        self._lock = threading.Lock()



    def analyze(self, img: np.ndarray, preferred_hand: int) -> tuple[str, float]:
        if img is None or img.size == 0:
            return "Unknown", 0.0

        processed = self._resize_for_inference(img)
        rgb_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        height, width = processed.shape[:2]

        with self._lock:
            hand_results = self._hands.process(rgb_img)
            # 同时执行 Face Mesh 推理
            face_mesh_results = self._face_mesh.process(rgb_img)

        direction = self._detect_direction(hand_results, preferred_hand)
        face_distance = self._estimate_distance_precise(face_mesh_results, width, height)
        return direction, face_distance

    @staticmethod
    def _resize_for_inference(img: np.ndarray, max_side: int = 480) -> np.ndarray:
        # 手机原图通常偏大，直接推理会增加延迟，这里先缩放到合适尺寸。
        height, width = img.shape[:2]
        longest_side = max(height, width)
        if longest_side <= max_side:
            return img
        scale = max_side / float(longest_side)
        target_size = (int(width * scale), int(height * scale))
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    def _detect_direction(self, hand_results, preferred_hand: int) -> str:
        if not hand_results.multi_hand_landmarks:
            return "Unknown"

        candidates = []
        for idx, handedness in enumerate(hand_results.multi_handedness):
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            candidates.append(
                {
                    "label": label,
                    "score": score,
                    "landmarks": hand_results.multi_hand_landmarks[idx],
                }
            )

        # 前端约定：0 表示优先识别右手，1 表示优先识别左手。
        # 如果目标手没有被识别到，则退化为使用置信度最高的手，
        # 避免整帧直接判定失败。
        target_label = "Right" if preferred_hand == 0 else "Left"
        selected = next(
            (item for item in sorted(candidates, key=lambda item: item["score"], reverse=True) if item["label"] == target_label),
            None,
        )
        if selected is None:
            selected = max(candidates, key=lambda item: item["score"])

        landmarks = selected["landmarks"]
        wrist = landmarks.landmark[self._hands_module.HandLandmark.WRIST]
        finger_tips = [
            landmarks.landmark[self._hands_module.HandLandmark.INDEX_FINGER_TIP],
            landmarks.landmark[self._hands_module.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks.landmark[self._hands_module.HandLandmark.RING_FINGER_TIP],
            landmarks.landmark[self._hands_module.HandLandmark.PINKY_TIP],
        ]
        avg_tip_x = sum(point.x for point in finger_tips) / len(finger_tips)
        avg_tip_y = sum(point.y for point in finger_tips) / len(finger_tips)

        # 通过“手腕 -> 四指平均指尖”的向量方向来判断手势朝向。
        dx = avg_tip_x - wrist.x
        dy = avg_tip_y - wrist.y

        if abs(dx) > abs(dy):
            return "Left" if dx < 0 else "Right"
        return "Up" if dy < 0 else "Down"

    def _estimate_distance_precise(self, face_mesh_results, image_width: int, image_height: int) -> float:
        """
        基于 Face Mesh 的眼角关键点计算距离（米）
        使用左眼外眼角 (landmark 33) 和右眼外眼角 (landmark 263)
        """
        if not face_mesh_results.multi_face_landmarks:
            return 0.0

        landmarks = face_mesh_results.multi_face_landmarks[0].landmark

        # 左眼外眼角 (index 33) 和 右眼外眼角 (index 263)
        left_eye_outer = (landmarks[33].x, landmarks[33].y)
        right_eye_outer = (landmarks[263].x, landmarks[263].y)

        # 计算图像中的像素距离（欧几里得距离）
        dx = (left_eye_outer[0] - right_eye_outer[0]) * image_width
        dy = (left_eye_outer[1] - right_eye_outer[1]) * image_height
        eye_pixel_distance = (dx ** 2 + dy ** 2) ** 0.5

        if eye_pixel_distance < 1e-6:
            return 0.0

        # 真实外眼角间距（米），成年人平均值约 0.10 米（10 厘米）
        real_eye_distance_m = 0.10
        # 焦距（像素），此处沿用经验值 800（可根据实际标定调整）
        focal_length = 800

        distance = (real_eye_distance_m * focal_length) / eye_pixel_distance
        return round(distance, 2)


app = Flask(__name__)
CORS(app)
vision_analyzer = VisionAnalyzer()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        # 每个请求上下文单独维护一个 SQLite 连接，避免跨线程复用连接。
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        g.db = connection
    return g.db


@app.teardown_appcontext
def close_db(_error):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db():
    # 服务启动时自动执行建表脚本，保证新环境可以直接启动。
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        connection.executescript(schema)
        connection.commit()


init_db()


def json_error(message: str, status: int = 400):
    return jsonify({"status": "failure", "message": message}), status


def get_authorization_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return None


def get_current_user():
    token = get_authorization_token()
    if not token:
        return None

    db = get_db()
    row = db.execute(
        """
        SELECT users.id, users.username, auth_tokens.token, auth_tokens.expires_at
        FROM auth_tokens
        JOIN users ON users.id = auth_tokens.user_id
        WHERE auth_tokens.token = ?
        """,
        (token,),
    ).fetchone()
    if row is None:
        return None

    # 过期 token 在访问时顺手清理。
    expires_at = datetime.fromisoformat(row["expires_at"])
    if expires_at <= utcnow():
        db.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
        db.commit()
        return None

    db.execute(
        "UPDATE auth_tokens SET last_used_at = ? WHERE token = ?",
        (utcnow_iso(), token),
    )
    db.commit()
    return row


def require_user():
    user = get_current_user()
    if user is None:
        abort(401, description="Authentication required")
    return user


def create_session(user_id: int) -> str:
    # 前端登录成功后会保存这个 bearer token，并在受保护接口里携带。
    db = get_db()
    token = secrets.token_urlsafe(32)
    expires_at = (utcnow() + timedelta(days=TOKEN_TTL_DAYS)).isoformat()
    db.execute(
        """
        INSERT INTO auth_tokens (token, user_id, expires_at, created_at, last_used_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (token, user_id, expires_at, utcnow_iso(), utcnow_iso()),
    )
    db.commit()
    return token


@app.errorhandler(401)
def unauthorized(error):
    return json_error(getattr(error, "description", "Unauthorized"), 401)


@app.errorhandler(404)
def not_found(error):
    return json_error(getattr(error, "description", "Not found"), 404)


@app.errorhandler(500)
def server_error(error):
    return json_error("Server error", 500)


@app.get("/")
def home():
    return jsonify(
        {
            "name": "visiontest-backend",
            "status": "ok",
            "version": "2.0.0",
            "time": utcnow_iso(),
        }
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "database": DATABASE_PATH.name,
            "mediapipe_solutions": bool(getattr(mp, "solutions", None)),
        }
    )


@app.post("/api/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    referral_code = (payload.get("referralCode") or "").strip().upper()

    if len(username) < 3:
        return json_error("Username must be at least 3 characters long")
    if len(password) < 6:
        return json_error("Password must be at least 6 characters long")
    if not referral_code:
        return json_error("Referral code is required")

    db = get_db()
    invite = db.execute(
        "SELECT code FROM invite_codes WHERE code = ? AND is_active = 1",
        (referral_code,),
    ).fetchone()
    if invite is None:
        return json_error("Referral code is invalid")

    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing is not None:
        return json_error("Username already exists")

    password_hash = generate_password_hash(password)
    cursor = db.execute(
        """
        INSERT INTO users (username, password_hash, referral_code_used, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (username, password_hash, referral_code, utcnow_iso()),
    )
    db.commit()
    token = create_session(cursor.lastrowid)
    return jsonify(
        {
            "status": "success",
            "token": token,
            "user": {"id": cursor.lastrowid, "username": username},
        }
    )


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""

    if not username or not password:
        return json_error("Username and password are required")

    db = get_db()
    user = db.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return json_error("Invalid username or password", 401)

    token = create_session(user["id"])
    return jsonify(
        {
            "status": "success",
            "token": token,
            "user": {"id": user["id"], "username": user["username"]},
        }
    )


@app.get("/api/auth/me")
def me():
    user = require_user()
    return jsonify(
        {
            "status": "success",
            "user": {"id": user["id"], "username": user["username"]},
        }
    )


@app.post("/api/auth/logout")
def logout():
    token = get_authorization_token()
    if not token:
        return jsonify({"status": "success"})
    db = get_db()
    db.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
    db.commit()
    return jsonify({"status": "success"})


@app.post("/api/upload")
def upload_image():
    # require_user()
    try:
        if "file" not in request.files:
            return json_error("No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return json_error("No file selected")

        # hand=0 表示优先按右手识别，hand=1 表示优先按左手识别。
        preferred_hand = request.headers.get("hand", default=0, type=int)

        # 直接在内存中解码上传图片，避免落临时文件。
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        image_bytes = in_memory_file.getvalue()
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return json_error("Image decoding failed")

        direction, face_distance = vision_analyzer.analyze(img, preferred_hand)
        return jsonify(
            {
                "status": "success",
                "direction": direction,
                "face_distance": face_distance,
            }
        )
    except Exception as exc:
        app.logger.exception("Upload analysis failed: %s", exc)
        return json_error("Image analysis failed", 500)


@app.post("/api/test-records")
def create_test_record():
    user = require_user()
    payload = request.get_json(silent=True) or {}

    # 后端存储的字段比当前历史页展示的字段更多，
    # 这样后面做统计或分析时不需要改写入协议。
    eye = (payload.get("eye") or "").strip().lower()
    result_label = str(payload.get("resultLabel") or "").strip()
    result_value = payload.get("resultValue")
    correct_count = int(payload.get("correctCount") or 0)
    wrong_count = int(payload.get("wrongCount") or 0)
    detected_distance = float(payload.get("detectedDistance") or 0)

    if eye not in {"left", "right"}:
        return json_error("Eye must be left or right")
    if not result_label:
        return json_error("Result label is required")
    try:
        result_value = float(result_value)
    except (TypeError, ValueError):
        return json_error("Result value must be numeric")

    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO vision_test_records (
            user_id,
            eye,
            result_label,
            result_value,
            correct_count,
            wrong_count,
            detected_distance,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
            eye,
            result_label,
            result_value,
            correct_count,
            wrong_count,
            detected_distance,
            utcnow_iso(),
        ),
    )
    db.commit()
    return jsonify({"status": "success", "recordId": cursor.lastrowid})


@app.get("/api/test-records")
def list_test_records():
    user = require_user()
    db = get_db()
    # 历史记录始终按当前登录用户隔离，前端只做展示层筛选，
    # 真正的数据归属限制在这里保证。
    rows = db.execute(
        """
        SELECT id, eye, result_label, result_value, correct_count, wrong_count, detected_distance, created_at
        FROM vision_test_records
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 100
        """,
        (user["id"],),
    ).fetchall()
    return jsonify(
        {
            "status": "success",
            "records": [dict(row) for row in rows],
        }
    )


@app.get("/api/download/guide")
def download_guide():
    chunk_size = 1024
    speed_limit = 50 * 1024

    try:
        if not GUIDE_PATH.is_file():
            abort(404, description="Guide file not found")

        def generate():
            with GUIDE_PATH.open("rb") as file_handle:
                while chunk := file_handle.read(chunk_size):
                    yield chunk
                    time.sleep(chunk_size / speed_limit)

        return Response(
            generate(),
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="guide.pdf"',
                "Content-Length": str(GUIDE_PATH.stat().st_size),
            },
        )
    except Exception as exc:
        app.logger.exception("Guide download failed: %s", exc)
        abort(500, description="Guide download failed")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8090)
