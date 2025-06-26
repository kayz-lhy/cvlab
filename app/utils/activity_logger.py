from flask import request
from app.models import UserActivity, db
from datetime import datetime

def log_user_activity(user_id, action, description=None, ip_address=None):
    """记录用户活动"""
    try:
        # 如果没有提供IP，尝试从请求中获取
        if ip_address is None:
            ip_address = get_client_ip()

        activity = UserActivity(
            user_id=user_id,
            action=action,
            description=description,
            ip_address=ip_address,
            created_at=datetime.utcnow()
        )

        db.session.add(activity)
        db.session.commit()

    except Exception as e:
        print(f"记录用户活动失败: {str(e)}")
        # 活动日志失败不应该影响主要功能
        try:
            db.session.rollback()
        except:
            pass

def get_client_ip():
    """获取客户端真实IP"""
    # 考虑代理和负载均衡器
    if request.headers.get('X-Forwarded-For'):
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers['X-Real-IP']
    else:
        return request.remote_addr

def log_login_activity(user_id, success=True, reason=None):
    """记录登录活动"""
    action = 'login_success' if success else 'login_failed'
    description = reason if not success else '用户成功登录'
    log_user_activity(user_id, action, description)

def log_logout_activity(user_id):
    """记录登出活动"""
    log_user_activity(user_id, 'logout', '用户登出')

def log_detection_activity(user_id, model_name, detection_count, processing_time):
    """记录检测活动"""
    description = f'使用模型 {model_name} 检测图像，发现 {detection_count} 个对象，耗时 {processing_time}s'
    log_user_activity(user_id, 'image_detection', description)

def log_user_management_activity(admin_id, action, target_user_id, details=None):
    """记录用户管理活动"""
    description = f'对用户 {target_user_id} 执行 {action} 操作'
    if details:
        description += f': {details}'
    log_user_activity(admin_id, f'manage_user_{action}', description)