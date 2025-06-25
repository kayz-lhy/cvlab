from flask import Blueprint, request, current_app, g
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from app.utils.response_helper import ResponseHelper
from app.utils.mongodb_helper import mongodb
from app.auth.decorators import token_required

# 创建检测蓝图
detection_bp = Blueprint('detection', __name__, url_prefix='/detection')

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

@detection_bp.route('/detect', methods=['POST'])
@token_required
def detect_image():
    """图片目标检测接口"""
    yolo_service = current_app.yolo_service

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

        # 构建响应数据
        response_data = {
            'image': result['result_image_base64'],
            'detections': formatted_detections,
            'processing_time': result['processing_time'],
            'detection_count': result['detection_count']
        }

        return ResponseHelper.success('图片检测完成', response_data)
    else:
        return ResponseHelper.error(f"检测失败: {result['error']}")

@detection_bp.route('/models', methods=['GET'])
@token_required
def get_models():
    """获取可用模型列表"""
    yolo_service = current_app.yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    models = yolo_service.list_available_weights()
    current_model = yolo_service.get_model_info()

    return ResponseHelper.success('获取模型列表成功', {
        'available_models': models,
        'current_model': current_model
    })

@detection_bp.route('/models/<model_name>/load', methods=['POST'])
@token_required
def load_model(model_name):
    """加载指定模型"""
    yolo_service = current_app.yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    success, message = yolo_service.load_model(model_name)

    if success:
        return ResponseHelper.success(message)
    else:
        return ResponseHelper.error(message)
