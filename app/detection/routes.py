from flask import Blueprint, request, current_app, g, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from app.services.yolo_service import YOLOv5Service
from app.utils.response_helper import ResponseHelper
from app.utils.mongodb_helper import mongodb
from app.auth.decorators import token_required

detection_bp = Blueprint('detection', __name__, url_prefix='/detection')

# 全局YOLOv5Service实例
yolo_service = None

def init_yolo_service(app):
    """初始化YOLO服务"""
    global yolo_service
    yolo_service = YOLOv5Service(
        yolo_repo_path=app.config['YOLO_REPO_PATH'],
        weights_path=app.config['YOLO_WEIGHTS_PATH']
    )

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

@detection_bp.route('/models/load', methods=['POST'])
@token_required
def load_model():
    """加载检测模型"""
    global yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    data = request.get_json()
    weights_filename = data.get('weights_filename', 'best.pt')

    success, message = yolo_service.load_model(weights_filename)

    if success:
        return ResponseHelper.success(message, {
            'model_info': yolo_service.get_model_info()
        })
    else:
        return ResponseHelper.error(message)

@detection_bp.route('/models/info', methods=['GET'])
@token_required
def get_model_info():
    """获取当前模型信息"""
    global yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    return ResponseHelper.success('获取模型信息成功', {
        'model_info': yolo_service.get_model_info(),
        'available_weights': yolo_service.list_available_weights()
    })

@detection_bp.route('/detect', methods=['POST'])
@token_required
def detect_image():
    """图片目标检测 - 主要接口"""
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
        result['record_id'] = record_id

        # 构建完整的结果数据
        response_data = {
            'detection_result': {
                'detections': result['detections'],
                'detection_count': result['detection_count'],
                'processing_time': result['processing_time'],
                'confidence_threshold': result['confidence_threshold'],
                'model_name': result['model_name']
            },
            'image_info': {
                'original_filename': file.filename,
                'image_size': result['image_size'],
                'file_size': result['file_size']
            },
            'result_image_base64': result['result_image_base64'],  # 直接返回base64
            'result_image_url': f"/detection/image/{os.path.basename(result['result_image_path'])}" if result['result_image_path'] else None,
            'record_id': record_id
        }

        return ResponseHelper.success('图片检测完成', response_data)
    else:
        return ResponseHelper.error(f"检测失败: {result['error']}")

@detection_bp.route('/image/<filename>')
def get_result_image(filename):
    """获取检测结果图片"""
    try:
        result_dir = current_app.config['RESULTS_FOLDER']
        file_path = os.path.join(result_dir, filename)

        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/jpeg')
        else:
            return ResponseHelper.not_found('图片不存在')
    except Exception as e:
        return ResponseHelper.error(f'获取图片失败: {str(e)}')

@detection_bp.route('/history', methods=['GET'])
@token_required
def get_detection_history():
    """获取用户检测历史"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    skip = (page - 1) * limit

    records = mongodb.get_user_detections(g.current_user.id, limit=limit, skip=skip)

    return ResponseHelper.success('获取检测历史成功', {
        'records': records,
        'page': page,
        'limit': limit,
        'total': len(records)
    })

@detection_bp.route('/batch', methods=['POST'])
@token_required
def batch_detect():
    """批量检测图片"""
    global yolo_service

    if not yolo_service or not yolo_service.model:
        return ResponseHelper.error('模型未加载，请先加载模型')

    files = request.files.getlist('images')
    if not files:
        return ResponseHelper.error('没有上传文件', 400)

    conf_threshold = float(request.form.get('confidence', 0.25))
    save_result = request.form.get('save_result', 'true').lower() == 'true'

    results = []
    successful_detections = 0

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        file_path = save_uploaded_file(file)
        if file_path:
            result = yolo_service.detect_image(
                image_path=file_path,
                conf_threshold=conf_threshold,
                save_result=save_result
            )

            if result['success']:
                # 保存检测记录到MongoDB
                record_id = mongodb.save_detection_record(g.current_user.id, result)
                result['record_id'] = record_id
                successful_detections += 1

            result['original_filename'] = file.filename
            results.append(result)

    if results:
        return ResponseHelper.success('批量检测完成', {
            'results': results,
            'total_files': len(results),
            'successful_detections': successful_detections
        })
    else:
        return ResponseHelper.error('没有有效的文件被处理')