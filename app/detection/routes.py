from flask import Blueprint, request, current_app, g, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from app.utils.response_helper import ResponseHelper
from app.utils.mongodb_helper import mongodb
from app.auth.decorators import token_required, admin_required
from app.services.yolo_service import ModelType
from app.utils.activity_logger import log_detection_activity, log_user_activity

# 创建检测蓝图
detection_bp = Blueprint('detection', __name__, url_prefix='/detection')

# 允许的图片文件扩展名
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
ALLOWED_WEIGHT_EXTENSIONS = {'pt', 'pth'}

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in allowed_extensions

def save_uploaded_file(file):
    """保存上传的文件"""
    if not file or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
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
        # 更新用户检测次数
        g.current_user.increment_detection_count()

        # 保存检测记录到MongoDB
        record_id = mongodb.save_detection_record(g.current_user.id, result)

        # 记录检测活动
        log_detection_activity(
            user_id=g.current_user.id,
            model_name=result.get('model_name', 'unknown'),
            detection_count=result.get('detection_count', 0),
            processing_time=result.get('processing_time', 0)
        )

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
            'detection_count': result['detection_count'],
            'model_info': {
                'model_name': result['model_name'],
                'model_type': result['model_type']
            }
        }

        return ResponseHelper.success('图片检测完成', response_data)
    else:
        return ResponseHelper.error(f"检测失败: {result['error']}")

@detection_bp.route('/models', methods=['GET'])
@token_required
def get_models():
    """获取可用模型列表（从MongoDB）"""
    yolo_service = current_app.yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    # 从MongoDB获取权重列表
    weights = yolo_service.list_available_weights()
    current_model = yolo_service.get_model_info()

    # 获取统计信息
    stats = mongodb.get_weight_statistics()

    return ResponseHelper.success('获取模型列表成功', {
        'available_weights': weights,
        'current_model': current_model,
        'statistics': stats
    })

@detection_bp.route('/models/<weight_id>/load', methods=['POST'])
@token_required
def load_model(weight_id):
    """从MongoDB加载指定模型"""
    yolo_service = current_app.yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    success, message = yolo_service.load_model_from_mongodb(weight_id)

    if success:
        # 记录模型切换活动
        model_info = yolo_service.get_model_info()
        log_user_activity(
            user_id=g.current_user.id,
            action='load_model',
            description=f'加载模型: {model_info.get("model_name", "unknown")}'
        )

        return ResponseHelper.success(message, {'model_info': model_info})
    else:
        return ResponseHelper.error(message)

@detection_bp.route('/weights/upload', methods=['POST'])
@token_required
@admin_required  # 只有管理员可以上传权重
def upload_weights():
    """上传权重文件到MongoDB（仅管理员）"""
    yolo_service = current_app.yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    # 检查是否有文件上传
    if 'weights_file' not in request.files:
        return ResponseHelper.error('没有上传权重文件', 400)

    file = request.files['weights_file']
    if file.filename == '':
        return ResponseHelper.error('没有选择文件', 400)

    if not allowed_file(file.filename, ALLOWED_WEIGHT_EXTENSIONS):
        return ResponseHelper.error('文件格式不支持，支持格式: pt, pth', 400)

    # 获取表单参数
    model_name = request.form.get('model_name', '').strip()
    description = request.form.get('description', '').strip()
    model_type_str = request.form.get('model_type', '').strip().lower()

    if not model_name:
        return ResponseHelper.error('模型名称不能为空', 400)

    # 解析模型类型
    model_type = None
    if model_type_str:
        try:
            if model_type_str in ['yolov5', 'v5']:
                model_type = ModelType.YOLOV5
            elif model_type_str in ['yolov8', 'v8']:
                model_type = ModelType.YOLOV8
            else:
                return ResponseHelper.error('不支持的模型类型，支持: yolov5, yolov8', 400)
        except:
            return ResponseHelper.error('模型类型格式错误', 400)

    # 保存临时文件
    temp_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        file.save(temp_path)

        # 上传到MongoDB
        result = yolo_service.upload_weights_to_mongodb(
            weights_file_path=temp_path,
            model_name=model_name,
            description=description,
            model_type=model_type
        )

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if result['success']:
            # 记录上传活动
            log_user_activity(
                user_id=g.current_user.id,
                action='upload_model',
                description=f'上传模型权重: {model_name} ({model_type.value if model_type else "auto"})'
            )

            return ResponseHelper.success('权重文件上传成功', result)
        else:
            return ResponseHelper.error(result['error'])

    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return ResponseHelper.error(f'上传失败: {str(e)}')

@detection_bp.route('/weights/<weight_id>', methods=['DELETE'])
@token_required
@admin_required  # 只有管理员可以删除权重
def delete_weights(weight_id):
    """删除权重文件（仅管理员）"""
    yolo_service = current_app.yolo_service

    if not yolo_service:
        return ResponseHelper.error('YOLO服务未初始化')

    success, message = yolo_service.delete_weight_from_mongodb(weight_id)

    if success:
        # 记录删除活动
        log_user_activity(
            user_id=g.current_user.id,
            action='delete_model',
            description=f'删除模型权重: {weight_id}'
        )
        return ResponseHelper.success(message)
    else:
        return ResponseHelper.error(message)

@detection_bp.route('/weights/<weight_id>/info', methods=['GET'])
@token_required
def get_weight_info(weight_id):
    """获取权重文件详细信息"""
    weight_record = mongodb.get_weight_by_id(weight_id)

    if not weight_record:
        return ResponseHelper.not_found('权重文件不存在')

    # 移除二进制数据，只返回元信息
    if 'weights_data' in weight_record:
        del weight_record['weights_data']

    return ResponseHelper.success('获取权重信息成功', weight_record)

@detection_bp.route('/weights/<weight_id>/update', methods=['PUT'])
@token_required
@admin_required  # 只有管理员可以更新权重信息
def update_weight_info(weight_id):
    """更新权重文件信息（仅管理员）"""
    data = request.get_json()

    if not data:
        return ResponseHelper.error('请提供更新数据', 400)

    # 只允许更新某些字段
    allowed_fields = ['model_name', 'description']
    updates = {k: v for k, v in data.items() if k in allowed_fields}

    if not updates:
        return ResponseHelper.error('没有可更新的字段', 400)

    success = mongodb.update_weight_info(weight_id, updates)

    if success:
        # 记录更新活动
        log_user_activity(
            user_id=g.current_user.id,
            action='update_model',
            description=f'更新模型权重信息: {weight_id}'
        )
        return ResponseHelper.success('权重信息更新成功')
    else:
        return ResponseHelper.error('权重信息更新失败')

@detection_bp.route('/model-types', methods=['GET'])
@token_required
def get_model_types():
    """获取支持的模型类型"""
    return ResponseHelper.success('获取模型类型成功', {
        'supported_types': [
            {
                'value': ModelType.YOLOV5.value,
                'label': 'YOLOv5',
                'description': '经典的YOLO模型，稳定可靠'
            },
            {
                'value': ModelType.YOLOV8.value,
                'label': 'YOLOv8',
                'description': '最新的YOLO模型，性能更优'
            }
        ]
    })

@detection_bp.route('/statistics', methods=['GET'])
@token_required
def get_detection_statistics():
    """获取检测统计信息"""
    user_id = None

    # 管理员可以查看所有用户的统计，普通用户只能查看自己的
    if g.current_user.is_admin():
        user_id = request.args.get('user_id', type=int)

    if not user_id:
        user_id = g.current_user.id

    # 获取检测统计
    detection_stats = mongodb.get_detection_statistics(user_id)

    # 获取权重统计（管理员才能看到）
    weight_stats = {}
    if g.current_user.is_admin():
        weight_stats = mongodb.get_weight_statistics()

    return ResponseHelper.success('获取统计信息成功', {
        'detection_statistics': detection_stats,
        'weight_statistics': weight_stats
    })

@detection_bp.route('/history', methods=['GET'])
@token_required
def get_detection_history():
    """获取检测历史记录"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    skip = (page - 1) * per_page

    # 普通用户只能查看自己的记录
    user_id = g.current_user.id

    # 管理员可以查看指定用户的记录
    if g.current_user.is_admin():
        requested_user_id = request.args.get('user_id', type=int)
        if requested_user_id:
            user_id = requested_user_id

    records = mongodb.get_user_detections(
        user_id=user_id,
        limit=per_page,
        skip=skip
    )

    return ResponseHelper.success('获取检测历史成功', {
        'records': records,
        'page': page,
        'per_page': per_page,
        'has_more': len(records) == per_page
    })