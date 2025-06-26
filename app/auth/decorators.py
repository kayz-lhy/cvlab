from functools import wraps
from flask import request, jsonify, g
from app.utils.jwt_helper import JWTHelper
from app.utils.response_helper import ResponseHelper
from app.models import User, UserActivity
from app.utils.activity_logger import log_user_activity

def token_required(f):
    """Token验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return ResponseHelper.unauthorized('Token缺失')

        if token.startswith('Bearer '):
            token = token[7:]

        user_id = JWTHelper.verify_token(token)
        if not user_id:
            return ResponseHelper.unauthorized('Token无效或已过期')

        user = User.query.get(user_id)
        if not user or not user.is_active:
            return ResponseHelper.unauthorized('用户不存在或已禁用')

        if user.is_locked():
            return ResponseHelper.unauthorized('账户已被锁定')

        g.current_user = user
        return f(*args, **kwargs)

    return decorated

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not hasattr(g, 'current_user'):
            return ResponseHelper.unauthorized('需要登录')
        return f(*args, **kwargs)

    return decorated

def admin_required(f):
    """管理员权限装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not hasattr(g, 'current_user'):
            return ResponseHelper.unauthorized('需要登录')

        if not g.current_user.is_admin():
            log_user_activity(
                user_id=g.current_user.id,
                action='admin_access_denied',
                description='尝试访问管理员功能'
            )
            return ResponseHelper.forbidden('需要管理员权限')

        return f(*args, **kwargs)
    return decorated

def self_or_admin_required(f):
    """自己或管理员权限装饰器（用户可以访问自己的资源，或者管理员可以访问所有资源）"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not hasattr(g, 'current_user'):
            return ResponseHelper.unauthorized('需要登录')

        # 获取URL中的user_id参数
        target_user_id = kwargs.get('user_id') or request.view_args.get('user_id')

        # 如果是管理员，允许访问
        if g.current_user.is_admin():
            return f(*args, **kwargs)

        # 如果是访问自己的资源，允许
        if target_user_id and int(target_user_id) == g.current_user.id:
            return f(*args, **kwargs)

        # 否则拒绝访问
        log_user_activity(
            user_id=g.current_user.id,
            action='access_denied',
            description=f'尝试访问用户 {target_user_id} 的资源'
        )
        return ResponseHelper.forbidden('只能访问自己的资源或需要管理员权限')

    return decorated