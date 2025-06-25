from flask import Blueprint, request, current_app, g
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from app.services.yolo_service import YOLOv5Service
from app.utils.response_helper import ResponseHelper
from app.utils.mongodb_helper import mongodb
from app.auth.decorators import token_required

# 创建专门为前端兼容的路由蓝图
detect_bp = Blueprint('detect', __name__, url_prefix='/detect')

# 使用detection模块的YOLO服务实例
from app.services import yolo_service

# 允许的图片文件扩展名
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS

def save_uploaded_file(file):
    """保存上传的文件"""
    if not file or not allowed_file(file.filename):
        return None

    # 创建上传目录
    upload_dir = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)

    # 生成唯一文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{timestamp}_{unique_id}{ext}"

    # 保存文件
    file_path = os.path.join(upload_dir, unique_filename)
    file.save(file_path)

    return file_path

@detect_bp.route('/detect', methods=['POST'])
@token_required
def detect_image():
    """图片目标检测 - 前端兼容接口"""
    global yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    if not yolo_service.model:
        return ResponseHelper.error('模型未加载，请先加载模型')

    # 检查是否有文件上传
    if 'image' not in request.files:
        return ResponseHelper.error('没有上传图片文件', 400)

    file = request.files['image']
    if file.filename == '':
        return ResponseHelper.error('没有选择文件', 400)

    # 获取参数
    conf_threshold = float(request.form.get('confidence', 0.25))
    save_result = request.form.get('save_result', 'true').lower() == 'true'

    # 保存上传的文件
    file_path = save_uploaded_file(file)
    if not file_path:
        return ResponseHelper.error('文件格式不支持，支持格式: png, jpg, jpeg, bmp, tiff, webp', 400)

    # 进行检测
    result = yolo_service.detect_image(
        image_path=file_path,
        conf_threshold=conf_threshold,
        save_result=save_result
    )

    if result['success']:
        # 保存检测记录到MongoDB
        record_id = mongodb.save_detection_record(g.current_user.id, result)

        # 转换检测结果为前端期望的格式
        formatted_detections = {}
        for detection in result['detections']:
            class_name = detection['class_name']
            if class_name not in formatted_detections:
                formatted_detections[class_name] = {
                    'count': 0,
                    'items': []
                }
            formatted_detections[class_name]['count'] += 1
            formatted_detections[class_name]['items'].append({
                'box': detection['bbox'],
                'confidence': detection['confidence'],
                'class': detection['class_name']
            })

        # 构建与前端完全兼容的响应数据
        response_data = {
            'image': result['result_image_base64'],  # 前端期望的图片字段
            'detections': formatted_detections       # 前端期望的检测结果格式
        }

        return ResponseHelper.success('图片检测完成', response_data)
    else:
        return ResponseHelper.error(f"检测失败: {result['error']}")