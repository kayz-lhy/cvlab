from functools import wraps
from flask import request, jsonify, g
from app.utils.jwt_helper import JWTHelper
from app.utils.response_helper import ResponseHelper
from app.models import User

def token_required(f):
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

        g.current_user = user
        return f(*args, **kwargs)

    return decorated

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not hasattr(g, 'current_user'):
            return ResponseHelper.unauthorized('需要登录')
        return f(*args, **kwargs)

    return decorated