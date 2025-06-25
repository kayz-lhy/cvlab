from flask import Blueprint, request, jsonify, g
from app.models import User, db
from app.utils.jwt_helper import JWTHelper
from app.utils.response_helper import ResponseHelper
from app.auth.decorators import token_required

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not all(k in data for k in ('username', 'email', 'password')):
        return ResponseHelper.error('缺少必要字段', 400)

    # 检查用户是否已存在
    if User.query.filter_by(username=data['username']).first():
        return ResponseHelper.conflict('用户名已存在')

    if User.query.filter_by(email=data['email']).first():
        return ResponseHelper.conflict('邮箱已存在')

    # 创建新用户
    user = User(
        username=data['username'],
        email=data['email']
    )
    user.set_password(data['password'])

    db.session.add(user)
    db.session.commit()

    return ResponseHelper.success('注册成功', {'user': user.to_dict()}, 201)

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not all(k in data for k in ('username', 'password')):
        return ResponseHelper.error('用户名和密码必须提供', 400)

    # 查找用户
    user = User.query.filter_by(username=data['username']).first()

    if not user or not user.check_password(data['password']):
        return ResponseHelper.unauthorized('用户名或密码错误')

    if not user.is_active:
        return ResponseHelper.unauthorized('账户已被禁用')

    # 生成token
    token = JWTHelper.generate_token(user.id)

    return ResponseHelper.success('登录成功', {
        'token': token,
        'user': user.to_dict()
    })

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout():
    # 撤销token
    JWTHelper.revoke_token(g.current_user.id)
    return ResponseHelper.success('登出成功')

@auth_bp.route('/profile', methods=['GET'])
@token_required
def get_profile():
    return ResponseHelper.success('获取用户信息成功', {
        'user': g.current_user.to_dict()
    })

@auth_bp.route('/refresh', methods=['POST'])
@token_required
def refresh_token():
    # 刷新token
    new_token = JWTHelper.refresh_token(g.current_user.id)
    return ResponseHelper.success('Token刷新成功', {
        'token': new_token
    })