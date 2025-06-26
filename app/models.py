from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from enum import Enum

# 这个db实例将在app/__init__.py中初始化
db = SQLAlchemy()

class UserRole(Enum):
    """用户角色枚举"""
    ADMIN = "admin"      # 管理员
    USER = "user"        # 普通用户

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # 用户基本信息
    full_name = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)

    # 角色和状态
    role = db.Column(db.Enum(UserRole), nullable=False, default=UserRole.USER)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)

    # 账户安全
    login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)

    # 使用统计
    detection_count = db.Column(db.Integer, default=0, nullable=False)

    def set_password(self, password):
        """设置密码"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        """检查是否是管理员"""
        return self.role == UserRole.ADMIN and self.is_active

    def is_locked(self):
        """检查账户是否被锁定"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until

    def lock_account(self, minutes=30):
        """锁定账户"""
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.login_attempts = 0
        db.session.commit()

    def unlock_account(self):
        """解锁账户"""
        self.locked_until = None
        self.login_attempts = 0
        db.session.commit()

    def increment_login_attempts(self):
        """增加登录尝试次数"""
        self.login_attempts += 1
        if self.login_attempts >= 5:  # 5次失败后锁定30分钟
            self.lock_account(30)
        db.session.commit()

    def reset_login_attempts(self):
        """重置登录尝试次数"""
        self.login_attempts = 0
        db.session.commit()

    def update_last_login(self):
        """更新最后登录时间"""
        self.last_login_at = datetime.utcnow()
        self.reset_login_attempts()
        db.session.commit()

    def increment_detection_count(self):
        """增加检测次数"""
        self.detection_count += 1
        db.session.commit()

    def to_dict(self, include_sensitive=False):
        """转换为字典"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'role': self.role.value,
            'role_name': self.get_role_name(),
            'is_admin': self.is_admin(),
            'is_active': self.is_active,
            'is_locked': self.is_locked(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'detection_count': self.detection_count
        }

        if include_sensitive:
            data.update({
                'login_attempts': self.login_attempts,
                'locked_until': self.locked_until.isoformat() if self.locked_until else None
            })

        return data

    def to_simple_dict(self):
        """简化的字典（用于列表显示）"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role.value,
            'role_name': self.get_role_name(),
            'is_admin': self.is_admin(),
            'is_active': self.is_active,
            'is_locked': self.is_locked(),
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'detection_count': self.detection_count
        }

    def get_role_name(self):
        """获取角色中文名称"""
        role_names = {
            UserRole.ADMIN: "管理员",
            UserRole.USER: "普通用户"
        }
        return role_names.get(self.role, "未知角色")

class UserActivity(db.Model):
    """用户活动日志"""
    __tablename__ = 'user_activities'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)  # login, logout, detect, manage_user等
    description = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    user = db.relationship('User', backref=db.backref('activities', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'action': self.action,
            'description': self.description,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat()
        }