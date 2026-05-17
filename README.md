# Backend

当前主后端文件：

- `server_v12.py`

技术栈：

- Flask
- Flask-CORS
- SQLite
- OpenCV
- MediaPipe
- NumPy

## 作用

后端负责：

- 用户注册、登录、退出
- token 鉴权
- 推荐码校验
- 图片上传
- 手势方向识别
- 人脸距离估计
- 视力检测记录写入与查询

## 运行环境

建议使用项目根目录现有虚拟环境：

- `c:\projects\visiontest-all\.venv`

如果需要自己重建环境，建议 Python 版本：

- Python 3.11 或与你当前虚拟环境一致的版本

说明：

- MediaPipe、TensorFlow、protobuf 的兼容性比较敏感。
- 不建议直接在系统 Python 里临时混装依赖。

## 安装依赖

如果你要在新的虚拟环境安装依赖：

```powershell
cd c:\projects\visiontest-all
.venv\Scripts\pip.exe install -r houduan\requirements.txt
```

## 启动方式

```powershell
c:\projects\visiontest-all\.venv\Scripts\python.exe c:\projects\visiontest-all\houduan\server_v12.py
```

默认监听：

- `http://0.0.0.0:8090`

## 数据库

当前使用 SQLite：

- 数据库文件：`houduan/vision_app.db`
- 建表 SQL：`houduan/schema.sql`

服务启动时会自动执行 `schema.sql`，因此新环境首次运行时会自动建表。

## 数据表

### `invite_codes`

推荐码表。

默认推荐码：

- `VISION2026`

### `users`

用户表，包含：

- 用户名
- 密码哈希
- 使用过的推荐码
- 创建时间

### `auth_tokens`

登录态 token 表。

### `vision_test_records`

检测历史记录表，包含：

- 用户 ID
- 左眼 / 右眼
- 视力标签
- 视力数值
- 正确次数
- 错误次数
- 检测距离
- 创建时间

## 测试数据

项目已提供种子脚本：

- `seed_test_data.py`

运行：

```powershell
c:\projects\visiontest-all\.venv\Scripts\python.exe c:\projects\visiontest-all\houduan\seed_test_data.py
```

默认会写入以下测试账号：

- `test_alice / Test123456`
- `test_bob / Test123456`
- `test_cindy / Test123456`

## 主要接口

### 健康检查

- `GET /api/health`

返回：

- 服务状态
- 当前数据库文件名
- `mediapipe.solutions` 是否可用

### 注册

- `POST /api/auth/register`

请求 JSON：

```json
{
  "username": "test001",
  "password": "123456",
  "referralCode": "VISION2026"
}
```

### 登录

- `POST /api/auth/login`

请求 JSON：

```json
{
  "username": "test001",
  "password": "123456"
}
```

### 获取当前用户

- `GET /api/auth/me`

请求头：

```text
Authorization: Bearer <token>
```

### 退出登录

- `POST /api/auth/logout`

### 上传检测图片

- `POST /api/upload`

请求要求：

- `multipart/form-data`
- 文件字段名：`file`
- 请求头里带 `Authorization`
- 请求头里带 `hand`

说明：

- `hand=0`：优先按右手识别
- `hand=1`：优先按左手识别

返回字段：

- `direction`
- `face_distance`

### 写入检测记录

- `POST /api/test-records`

请求 JSON 示例：

```json
{
  "eye": "left",
  "resultLabel": "4.8",
  "resultValue": 4.8,
  "correctCount": 2,
  "wrongCount": 0,
  "detectedDistance": 1.41
}
```

### 查询历史记录

- `GET /api/test-records`

说明：

- 只返回当前登录用户自己的记录
- 后端按时间倒序返回最近 100 条

## 常见问题

### 1. `module 'mediapipe' has no attribute 'solutions'`

说明当前启动环境不对，或 MediaPipe 安装不完整。

优先使用：

```powershell
c:\projects\visiontest-all\.venv\Scripts\python.exe c:\projects\visiontest-all\houduan\server_v12.py
```

### 2. protobuf / tensorflow 导入报错

这通常是系统 Python 环境里安装了不兼容版本。

建议：

1. 不要混用系统 Python 和项目虚拟环境
2. 固定使用 `.venv\Scripts\python.exe`

### 3. 前端可以打开但接口全失败

优先检查：

1. 后端是否已经启动
2. 前端请求的后端地址是否正确
3. 浏览器控制台 / 网络面板里是否有 401、500、CORS 报错

## 关键文件

- `server_v12.py`：主服务
- `schema.sql`：建表 SQL
- `seed_test_data.py`：测试数据脚本
- `requirements.txt`：依赖列表
