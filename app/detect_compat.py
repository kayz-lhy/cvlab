from flask import Blueprint
from app.detection.routes import detect_image

# 创建前端兼容蓝图，将 /detect/detect 路由映射到检测功能
detect_compat_bp = Blueprint('detect_compat', __name__, url_prefix='/detect')

# 直接使用detection模块的检测函数
detect_compat_bp.add_url_rule('/detect', 'detect_image', detect_image, methods=['POST'])