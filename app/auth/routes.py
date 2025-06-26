from flask import Blueprint, request, jsonify, g
from app.models import User, db, UserRole
from app.utils.jwt_helper import JWTHelper
from app.utils.response_helper import ResponseHelper
from app.auth.decorators import token_required
from app.utils.activity_logger import log_login_activity, log_logout_activity, log_user_activity
from datetime import datetime

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        data = request.get_json()

        if not data or not all(k in data for k in ('username', 'email', 'password')):
            return ResponseHelper.error('缺少必要字段: username, email, password', 400)

        # 验证密码强度
        password = data['password']
        if len(password) < 6:
            return ResponseHelper.error('密码长度至少6位', 400)

        # 检查用户是否已存在
        if User.query.filter_by(username=data['username']).first():
            return ResponseHelper.conflict('用户名已存在')

        if User.query.filter_by(email=data['email']).first():
            return ResponseHelper.conflict('邮箱已存在')

        # 创建新用户（默认为普通用户）
        user = User(
            username=data['username'],
            email=data['email'],
            full_name=data.get('full_name', ''),
            phone=data.get('phone', ''),
            role=UserRole.USER  # 默认普通用户
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        # 记录注册活动
        log_user_activity(
            user_id=user.id,
            action='register',
            description='用户注册成功'
        )

        return ResponseHelper.success('注册成功', {
            'user': user.to_dict()
        }, 201)

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'注册失败: {str(e)}')

@auth_bp.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()

        if not data or not all(k in data for k in ('username', 'password')):
            return ResponseHelper.error('用户名和密码必须提供', 400)

        # 查找用户
        user = User.query.filter_by(username=data['username']).first()

        if not user:
            # 记录失败的登录尝试
            log_user_activity(
                user_id=None,
                action='login_failed',
                description=f'登录失败: 用户名 {data["username"]} 不存在'
            )
            return ResponseHelper.unauthorized('用户名或密码错误')

        # 检查账户是否被锁定
        if user.is_locked():
            log_login_activity(user.id, False, '账户已被锁定')
            return ResponseHelper.unauthorized('账户已被锁定，请稍后再试')

        # 检查密码
        if not user.check_password(data['password']):
            # 增加失败登录次数
            user.increment_login_attempts()
            log_login_activity(user.id, False, '密码错误')

            remaining_attempts = 5 - user.login_attempts
            if remaining_attempts > 0:
                return ResponseHelper.unauthorized(f'用户名或密码错误，剩余尝试次数: {remaining_attempts}')
            else:
                return ResponseHelper.unauthorized('登录失败次数过多，账户已被锁定30分钟')

        # 检查账户状态
        if not user.is_active:
            log_login_activity(user.id, False, '账户已被禁用')
            return ResponseHelper.unauthorized('账户已被禁用')

        # 登录成功
        user.update_last_login()

        # 生成token
        token = JWTHelper.generate_token(user.id)

        # 记录成功登录
        log_login_activity(user.id, True)

        return ResponseHelper.success('登录成功', {
            'token': token,
            'user': user.to_dict()
        })

    except Exception as e:
        return ResponseHelper.error(f'登录失败: {str(e)}')

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout():
    """用户登出"""
    try:
        # 撤销token
        JWTHelper.revoke_token(g.current_user.id)

        # 记录登出活动
        log_logout_activity(g.current_user.id)

        return ResponseHelper.success('登出成功')
    except Exception as e:
        return ResponseHelper.error(f'登出失败: {str(e)}')

@auth_bp.route('/profile', methods=['GET'])
@token_required
def get_profile():
    """获取当前用户信息"""
    try:
        return ResponseHelper.success('获取用户信息成功', {
            'user': g.current_user.to_dict()
        })
    except Exception as e:
        return ResponseHelper.error(f'获取用户信息失败: {str(e)}')

@auth_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile():
    """更新当前用户信息"""
    try:
        data = request.get_json()
        if not data:
            return ResponseHelper.error('没有提供更新数据')

        # 允许用户更新的字段
        allowed_fields = ['full_name', 'phone']
        updated_fields = []

        for field in allowed_fields:
            if field in data:
                old_value = getattr(g.current_user, field)
                new_value = data[field]
                if old_value != new_value:
                    setattr(g.current_user, field, new_value)
                    updated_fields.append(f'{field}: {old_value} -> {new_value}')

        if updated_fields:
            g.current_user.updated_at = datetime.utcnow()
            db.session.commit()

            # 记录活动
            log_user_activity(
                user_id=g.current_user.id,
                action='update_profile',
                description=f'更新个人信息: {", ".join(updated_fields)}'
            )

        return ResponseHelper.success('个人信息更新成功', {
            'user': g.current_user.to_dict(),
            'updated_fields': updated_fields
        })

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'更新个人信息失败: {str(e)}')

@auth_bp.route('/change-password', methods=['POST'])
@token_required
def change_password():
    """修改密码"""
    try:
        data = request.get_json()

        if not data or not all(k in data for k in ('old_password', 'new_password')):
            return ResponseHelper.error('需要提供旧密码和新密码')

        # 验证旧密码
        if not g.current_user.check_password(data['old_password']):
            return ResponseHelper.error('旧密码错误')

        # 验证新密码强度
        new_password = data['new_password']
        if len(new_password) < 6:
            return ResponseHelper.error('新密码长度至少6位')

        if new_password == data['old_password']:
            return ResponseHelper.error('新密码不能与旧密码相同')

        # 更新密码
        g.current_user.set_password(new_password)
        g.current_user.updated_at = datetime.utcnow()
        db.session.commit()

        # 记录活动
        log_user_activity(
            user_id=g.current_user.id,
            action='change_password',
            description='用户修改密码'
        )

        return ResponseHelper.success('密码修改成功')

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'修改密码失败: {str(e)}')

@auth_bp.route('/refresh', methods=['POST'])
@token_required
def refresh_token():
    """刷新token"""
    try:
        # 刷新token
        new_token = JWTHelper.refresh_token(g.current_user.id)

        # 记录活动
        log_user_activity(
            user_id=g.current_user.id,
            action='refresh_token',
            description='刷新访问令牌'
        )

        return ResponseHelper.success('Token刷新成功', {
            'token': new_token,
            'user': g.current_user.to_dict()
        })
    except Exception as e:
        return ResponseHelper.error(f'Token刷新失败: {str(e)}')

@auth_bp.route('/my-activities', methods=['GET'])
@token_required
def get_my_activities():
    """获取当前用户的活动记录"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 50)
        action_filter = request.args.get('action')

        from app.models import UserActivity
        from datetime import timedelta

        # 构建查询
        query = UserActivity.query.filter_by(user_id=g.current_user.id)

        # 只显示最近30天的记录
        start_date = datetime.utcnow() - timedelta(days=30)
        query = query.filter(UserActivity.created_at >= start_date)

        # 动作筛选
        if action_filter:
            query = query.filter(UserActivity.action.ilike(f'%{action_filter}%'))

        # 排序和分页
        query = query.order_by(UserActivity.created_at.desc())
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        activities = []
        for activity in pagination.items:
            activity_dict = activity.to_dict()
            # 隐藏敏感信息
            activity_dict.pop('ip_address', None)
            activities.append(activity_dict)

        return ResponseHelper.success('获取活动记录成功', {
            'activities': activities,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_prev': pagination.has_prev,
                'has_next': pagination.has_next
            }
        })

    except Exception as e:
        return ResponseHelper.error(f'获取活动记录失败: {str(e)}')