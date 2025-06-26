from flask import Blueprint, request, g
from app.models import User, UserActivity, UserRole, db
from app.utils.response_helper import ResponseHelper
from app.auth.decorators import token_required, admin_required, self_or_admin_required
from app.utils.activity_logger import log_user_management_activity, log_user_activity
from datetime import datetime, timedelta
from sqlalchemy import or_

user_mgmt_bp = Blueprint('user_management', __name__, url_prefix='/admin/users')

@user_mgmt_bp.route('/list', methods=['GET'])
@token_required
@admin_required
def list_users():
    """获取用户列表（仅管理员）"""
    try:
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        role = request.args.get('role')
        is_active = request.args.get('is_active')
        search = request.args.get('search', '').strip()

        # 构建查询
        query = User.query

        # 角色筛选
        if role:
            try:
                role_enum = UserRole(role)
                query = query.filter(User.role == role_enum)
            except ValueError:
                return ResponseHelper.error('无效的角色参数')

        # 活跃状态筛选
        if is_active is not None:
            active_bool = is_active.lower() in ['true', '1', 'yes']
            query = query.filter(User.is_active == active_bool)

        # 搜索筛选
        if search:
            search_pattern = f'%{search}%'
            query = query.filter(or_(
                User.username.ilike(search_pattern),
                User.email.ilike(search_pattern),
                User.full_name.ilike(search_pattern)
            ))

        # 排序
        query = query.order_by(User.created_at.desc())

        # 分页
        pagination = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )

        users = [user.to_simple_dict() for user in pagination.items]

        # 记录活动
        log_user_activity(
            user_id=g.current_user.id,
            action='list_users',
            description=f'查看用户列表，页码: {page}'
        )

        return ResponseHelper.success('获取用户列表成功', {
            'users': users,
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
        return ResponseHelper.error(f'获取用户列表失败: {str(e)}')

@user_mgmt_bp.route('/<int:user_id>', methods=['GET'])
@token_required
@self_or_admin_required
def get_user(user_id):
    """获取用户详情"""
    try:
        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        # 管理员可以看到敏感信息
        include_sensitive = g.current_user.is_admin() and g.current_user.id != user_id

        # 记录活动
        log_user_activity(
            user_id=g.current_user.id,
            action='view_user',
            description=f'查看用户 {user.username} ({user_id}) 的详情'
        )

        return ResponseHelper.success('获取用户详情成功', {
            'user': user.to_dict(include_sensitive=include_sensitive)
        })

    except Exception as e:
        return ResponseHelper.error(f'获取用户详情失败: {str(e)}')

@user_mgmt_bp.route('/create', methods=['POST'])
@token_required
@admin_required
def create_user():
    """创建用户（仅管理员）"""
    try:
        data = request.get_json()

        # 验证必要字段
        required_fields = ['username', 'email', 'password']
        if not data or not all(field in data for field in required_fields):
            return ResponseHelper.error('缺少必要字段: username, email, password')

        # 检查用户是否已存在
        if User.query.filter_by(username=data['username']).first():
            return ResponseHelper.conflict('用户名已存在')

        if User.query.filter_by(email=data['email']).first():
            return ResponseHelper.conflict('邮箱已存在')

        # 验证角色
        role = UserRole.USER  # 默认角色
        if 'role' in data:
            try:
                role = UserRole(data['role'])
            except ValueError:
                return ResponseHelper.error('无效的角色，只能是 admin 或 user')

        # 创建用户
        user = User(
            username=data['username'],
            email=data['email'],
            full_name=data.get('full_name', ''),
            phone=data.get('phone', ''),
            role=role,
            is_active=data.get('is_active', True)
        )
        user.set_password(data['password'])

        db.session.add(user)
        db.session.commit()

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='create',
            target_user_id=user.id,
            details=f'创建用户 {user.username}，角色: {role.value}'
        )

        return ResponseHelper.success('用户创建成功', {
            'user': user.to_dict()
        }, 201)

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'创建用户失败: {str(e)}')

@user_mgmt_bp.route('/<int:user_id>/update', methods=['PUT'])
@token_required
@self_or_admin_required
def update_user(user_id):
    """更新用户信息"""
    try:
        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        data = request.get_json()
        if not data:
            return ResponseHelper.error('没有提供更新数据')

        updated_fields = []

        # 如果是更新自己的信息，限制可更新字段
        if g.current_user.id == user_id:
            # 普通用户只能更新基本信息
            for field in ['full_name', 'phone']:
                if field in data:
                    old_value = getattr(user, field)
                    new_value = data[field]
                    if old_value != new_value:
                        setattr(user, field, new_value)
                        updated_fields.append(f'{field}: {old_value} -> {new_value}')
        else:
            # 管理员可以更新更多字段
            if not g.current_user.is_admin():
                return ResponseHelper.forbidden('只能修改自己的信息')

            # 更新基本信息
            for field in ['full_name', 'phone']:
                if field in data:
                    old_value = getattr(user, field)
                    new_value = data[field]
                    if old_value != new_value:
                        setattr(user, field, new_value)
                        updated_fields.append(f'{field}: {old_value} -> {new_value}')

            # 更新用户名和邮箱（需要检查唯一性）
            if 'username' in data and data['username'] != user.username:
                if User.query.filter(User.username == data['username'], User.id != user_id).first():
                    return ResponseHelper.conflict('用户名已存在')
                old_username = user.username
                user.username = data['username']
                updated_fields.append(f'username: {old_username} -> {data["username"]}')

            if 'email' in data and data['email'] != user.email:
                if User.query.filter(User.email == data['email'], User.id != user_id).first():
                    return ResponseHelper.conflict('邮箱已存在')
                old_email = user.email
                user.email = data['email']
                updated_fields.append(f'email: {old_email} -> {data["email"]}')

            # 更新角色
            if 'role' in data:
                try:
                    new_role = UserRole(data['role'])
                    if new_role != user.role:
                        # 不能修改最后一个管理员的角色
                        if user.role == UserRole.ADMIN and new_role != UserRole.ADMIN:
                            admin_count = User.query.filter_by(role=UserRole.ADMIN, is_active=True).count()
                            if admin_count <= 1:
                                return ResponseHelper.error('不能修改最后一个管理员的角色')

                        old_role = user.role.value
                        user.role = new_role
                        updated_fields.append(f'role: {old_role} -> {new_role.value}')
                except ValueError:
                    return ResponseHelper.error('无效的角色，只能是 admin 或 user')

            # 更新状态
            if 'is_active' in data:
                new_active = bool(data['is_active'])
                if user.is_active != new_active:
                    # 不能禁用最后一个管理员
                    if not new_active and user.role == UserRole.ADMIN:
                        admin_count = User.query.filter_by(role=UserRole.ADMIN, is_active=True).count()
                        if admin_count <= 1:
                            return ResponseHelper.error('不能禁用最后一个管理员')

                    user.is_active = new_active
                    updated_fields.append(f'is_active: {not new_active} -> {new_active}')

        # 更新密码
        if 'password' in data:
            if g.current_user.id == user_id:
                # 用户修改自己的密码需要提供旧密码
                if 'old_password' not in data:
                    return ResponseHelper.error('修改密码需要提供旧密码')
                if not user.check_password(data['old_password']):
                    return ResponseHelper.error('旧密码错误')
            elif not g.current_user.is_admin():
                return ResponseHelper.forbidden('无权修改其他用户密码')

            user.set_password(data['password'])
            updated_fields.append('password: ****')

        if not updated_fields:
            return ResponseHelper.success('没有字段需要更新')

        user.updated_at = datetime.utcnow()
        db.session.commit()

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='update',
            target_user_id=user.id,
            details=f'更新字段: {", ".join(updated_fields)}'
        )

        return ResponseHelper.success('用户信息更新成功', {
            'user': user.to_dict(),
            'updated_fields': updated_fields
        })

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'更新用户信息失败: {str(e)}')

@user_mgmt_bp.route('/<int:user_id>/delete', methods=['DELETE'])
@token_required
@admin_required
def delete_user(user_id):
    """删除用户（仅管理员，软删除）"""
    try:
        if user_id == g.current_user.id:
            return ResponseHelper.error('不能删除自己')

        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        # 不能删除最后一个管理员
        if user.role == UserRole.ADMIN:
            admin_count = User.query.filter_by(role=UserRole.ADMIN, is_active=True).count()
            if admin_count <= 1:
                return ResponseHelper.error('不能删除最后一个管理员')

        # 软删除（设置为非活跃）
        old_username = user.username
        user.is_active = False
        user.username = f"{user.username}_deleted_{int(datetime.utcnow().timestamp())}"
        user.email = f"{user.email}_deleted_{int(datetime.utcnow().timestamp())}"
        user.updated_at = datetime.utcnow()

        db.session.commit()

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='delete',
            target_user_id=user.id,
            details=f'删除用户 {old_username}'
        )

        return ResponseHelper.success('用户删除成功')

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'删除用户失败: {str(e)}')

@user_mgmt_bp.route('/<int:user_id>/lock', methods=['POST'])
@token_required
@admin_required
def lock_user(user_id):
    """锁定用户（仅管理员）"""
    try:
        if user_id == g.current_user.id:
            return ResponseHelper.error('不能锁定自己')

        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        data = request.get_json() or {}
        minutes = data.get('minutes', 30)  # 默认锁定30分钟

        user.lock_account(minutes)

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='lock',
            target_user_id=user.id,
            details=f'锁定用户 {minutes} 分钟'
        )

        return ResponseHelper.success(f'用户已锁定 {minutes} 分钟')

    except Exception as e:
        return ResponseHelper.error(f'锁定用户失败: {str(e)}')

@user_mgmt_bp.route('/<int:user_id>/unlock', methods=['POST'])
@token_required
@admin_required
def unlock_user(user_id):
    """解锁用户（仅管理员）"""
    try:
        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        user.unlock_account()

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='unlock',
            target_user_id=user.id,
            details='解锁用户账户'
        )

        return ResponseHelper.success('用户已解锁')

    except Exception as e:
        return ResponseHelper.error(f'解锁用户失败: {str(e)}')

@user_mgmt_bp.route('/<int:user_id>/activities', methods=['GET'])
@token_required
@self_or_admin_required
def get_user_activities(user_id):
    """获取用户活动记录"""
    try:
        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        action_filter = request.args.get('action')
        days = request.args.get('days', 30, type=int)  # 默认查看30天内的记录

        # 构建查询
        query = UserActivity.query.filter_by(user_id=user_id)

        # 时间筛选
        if days > 0:
            start_date = datetime.utcnow() - timedelta(days=days)
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
            # 如果不是管理员且不是查看自己的记录，隐藏IP地址
            if not g.current_user.is_admin() and g.current_user.id != user_id:
                activity_dict.pop('ip_address', None)
            activities.append(activity_dict)

        # 记录活动
        log_user_activity(
            user_id=g.current_user.id,
            action='view_activities',
            description=f'查看用户 {user.username} 的活动记录'
        )

        return ResponseHelper.success('获取用户活动记录成功', {
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
        return ResponseHelper.error(f'获取用户活动记录失败: {str(e)}')

@user_mgmt_bp.route('/statistics', methods=['GET'])
@token_required
@admin_required
def get_user_statistics():
    """获取用户统计信息（仅管理员）"""
    try:
        # 基本统计
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        admin_users = User.query.filter_by(role=UserRole.ADMIN, is_active=True).count()
        normal_users = User.query.filter_by(role=UserRole.USER, is_active=True).count()
        locked_users = User.query.filter(User.locked_until > datetime.utcnow()).count()

        # 最近注册用户（7天内）
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_registrations = User.query.filter(User.created_at >= week_ago).count()

        # 最近活跃用户（7天内有登录）
        recent_active = User.query.filter(
            User.last_login_at >= week_ago,
            User.is_active == True
        ).count()

        # 活动统计
        total_activities = UserActivity.query.count()
        recent_activities = UserActivity.query.filter(
            UserActivity.created_at >= week_ago
        ).count()

        # 记录活动
        log_user_activity(
            user_id=g.current_user.id,
            action='view_statistics',
            description='查看用户统计信息'
        )

        return ResponseHelper.success('获取用户统计成功', {
            'total_users': total_users,
            'active_users': active_users,
            'admin_users': admin_users,
            'normal_users': normal_users,
            'locked_users': locked_users,
            'recent_registrations': recent_registrations,
            'recent_active': recent_active,
            'total_activities': total_activities,
            'recent_activities': recent_activities
        })

    except Exception as e:
        return ResponseHelper.error(f'获取用户统计失败: {str(e)}')

@user_mgmt_bp.route('/make-admin/<int:user_id>', methods=['POST'])
@token_required
@admin_required
def make_admin(user_id):
    """设置用户为管理员（仅管理员）"""
    try:
        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        if user.role == UserRole.ADMIN:
            return ResponseHelper.success('用户已经是管理员')

        old_role = user.role.value
        user.role = UserRole.ADMIN
        user.updated_at = datetime.utcnow()
        db.session.commit()

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='promote_to_admin',
            target_user_id=user.id,
            details=f'将用户从 {old_role} 提升为管理员'
        )

        return ResponseHelper.success(f'用户 {user.username} 已设置为管理员')

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'设置管理员失败: {str(e)}')

@user_mgmt_bp.route('/remove-admin/<int:user_id>', methods=['POST'])
@token_required
@admin_required
def remove_admin(user_id):
    """移除用户的管理员权限（仅管理员）"""
    try:
        if user_id == g.current_user.id:
            return ResponseHelper.error('不能移除自己的管理员权限')

        user = User.query.get(user_id)
        if not user:
            return ResponseHelper.not_found('用户不存在')

        if user.role != UserRole.ADMIN:
            return ResponseHelper.success('用户不是管理员')

        # 检查是否是最后一个管理员
        admin_count = User.query.filter_by(role=UserRole.ADMIN, is_active=True).count()
        if admin_count <= 1:
            return ResponseHelper.error('不能移除最后一个管理员的权限')

        old_role = user.role.value
        user.role = UserRole.USER
        user.updated_at = datetime.utcnow()
        db.session.commit()

        # 记录活动
        log_user_management_activity(
            admin_id=g.current_user.id,
            action='remove_admin',
            target_user_id=user.id,
            details=f'将用户从 {old_role} 降级为普通用户'
        )

        return ResponseHelper.success(f'已移除用户 {user.username} 的管理员权限')

    except Exception as e:
        db.session.rollback()
        return ResponseHelper.error(f'移除管理员权限失败: {str(e)}')