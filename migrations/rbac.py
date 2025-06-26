"""
数据库迁移脚本：从原有用户表升级到简化RBAC权限系统

使用方法：
1. 确保已安装Flask-Migrate: pip install Flask-Migrate
2. 运行迁移命令:
   flask db init (如果是第一次)
   flask db migrate -m "upgrade to simple rbac"
   flask db upgrade
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

def upgrade():
    """升级数据库结构"""

    # 检查users表是否存在，如果不存在则创建
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if 'users' not in inspector.get_table_names():
        # 创建新的users表
        op.create_table('users',
                        sa.Column('id', sa.Integer(), nullable=False),
                        sa.Column('username', sa.String(length=80), nullable=False),
                        sa.Column('email', sa.String(length=120), nullable=False),
                        sa.Column('password_hash', sa.String(length=255), nullable=False),
                        sa.Column('full_name', sa.String(length=120), nullable=True),
                        sa.Column('phone', sa.String(length=20), nullable=True),
                        sa.Column('role', sa.Enum('ADMIN', 'USER', name='userrole'), nullable=False),
                        sa.Column('is_active', sa.Boolean(), nullable=False),
                        sa.Column('created_at', sa.DateTime(), nullable=False),
                        sa.Column('updated_at', sa.DateTime(), nullable=True),
                        sa.Column('last_login_at', sa.DateTime(), nullable=True),
                        sa.Column('login_attempts', sa.Integer(), nullable=False),
                        sa.Column('locked_until', sa.DateTime(), nullable=True),
                        sa.Column('detection_count', sa.Integer(), nullable=False),
                        sa.PrimaryKeyConstraint('id')
                        )

        # 创建索引
        op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
        op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    else:
        # 检查并添加缺失的列
        columns = [col['name'] for col in inspector.get_columns('users')]

        # 添加新字段（如果不存在）
        if 'full_name' not in columns:
            op.add_column('users', sa.Column('full_name', sa.String(length=120), nullable=True))

        if 'phone' not in columns:
            op.add_column('users', sa.Column('phone', sa.String(length=20), nullable=True))

        if 'role' not in columns:
            # 先创建enum类型
            role_enum = sa.Enum('ADMIN', 'USER', name='userrole')
            role_enum.create(conn, checkfirst=True)

            # 添加role列，默认为USER
            op.add_column('users', sa.Column('role', role_enum, nullable=False, server_default='USER'))

        if 'login_attempts' not in columns:
            op.add_column('users', sa.Column('login_attempts', sa.Integer(), nullable=False, server_default='0'))

        if 'locked_until' not in columns:
            op.add_column('users', sa.Column('locked_until', sa.DateTime(), nullable=True))

        if 'detection_count' not in columns:
            op.add_column('users', sa.Column('detection_count', sa.Integer(), nullable=False, server_default='0'))

        if 'updated_at' not in columns:
            op.add_column('users', sa.Column('updated_at', sa.DateTime(), nullable=True))

        if 'last_login_at' not in columns:
            op.add_column('users', sa.Column('last_login_at', sa.DateTime(), nullable=True))

    # 创建用户活动表
    if 'user_activities' not in inspector.get_table_names():
        op.create_table('user_activities',
                        sa.Column('id', sa.Integer(), nullable=False),
                        sa.Column('user_id', sa.Integer(), nullable=False),
                        sa.Column('action', sa.String(length=100), nullable=False),
                        sa.Column('description', sa.Text(), nullable=True),
                        sa.Column('ip_address', sa.String(length=45), nullable=True),
                        sa.Column('created_at', sa.DateTime(), nullable=False),
                        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
                        sa.PrimaryKeyConstraint('id')
                        )

def downgrade():
    """降级数据库结构（谨慎使用）"""

    # 删除用户活动表
    op.drop_table('user_activities')

    # 注意：这里不删除users表，因为可能包含重要数据
    # 如果需要完全回滚，请手动处理

def create_default_admin():
    """创建默认管理员账户的独立函数"""
    from app.models import User, UserRole, db
    from werkzeug.security import generate_password_hash

    try:
        # 检查是否已有管理员
        admin_exists = User.query.filter_by(role=UserRole.ADMIN).first()

        if not admin_exists:
            # 创建默认管理员
            admin = User(
                username='admin',
                email='admin@cvlab.com',
                full_name='系统管理员',
                role=UserRole.ADMIN,
                is_active=True,
                login_attempts=0,
                detection_count=0
            )
            admin.set_password('admin123')  # 默认密码

            db.session.add(admin)
            db.session.commit()

            print("✅ 默认管理员账户创建成功")
            print("📝 管理员登录信息: username=admin, password=admin123")
            print("⚠️ 请在首次登录后立即修改默认密码")

    except Exception as e:
        print(f"❌ 创建默认管理员失败: {str(e)}")
        db.session.rollback()