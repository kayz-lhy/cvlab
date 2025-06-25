from flask import Flask
from flask_migrate import Migrate
from flask_cors import CORS
from config import config
from app.utils.redis_helper import init_redis
from app.utils.mongodb_helper import mongodb
from app.models import db
import os
import sys

migrate = Migrate()

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # 创建必要的目录
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'app/static/uploads'), exist_ok=True)
    os.makedirs(app.config.get('RESULTS_FOLDER', 'app/static/results'), exist_ok=True)
    os.makedirs(app.config.get('YOLO_WEIGHTS_PATH', 'weights'), exist_ok=True)

    # 调试：检查文件和路径
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"📁 Python路径: {sys.path[:3]}...")
    print(f"📄 yolo_service.py存在: {os.path.exists('app/services/yolo_service.py')}")
    print(f"📄 yolov5目录存在: {os.path.exists('yolov5')}")
    print(f"📄 weights目录存在: {os.path.exists('weights')}")

    # 初始化数据库
    db.init_app(app)

    # 初始化数据库迁移
    migrate.init_app(app, db)

    # 初始化CORS
    CORS(app,
         origins=app.config.get('CORS_ORIGINS', ['*']),
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         supports_credentials=True)

    # 初始化Redis
    init_redis(app)

    # 初始化MongoDB
    mongodb.init_app(app)

    # 初始化YOLO服务
    try:
        from app.detection.routes import init_yolo_service
        init_yolo_service(app)
        print("✅ YOLO服务初始化成功")
    except Exception as e:
        print(f"❌ YOLO服务初始化失败: {str(e)}")

    # 注册蓝图
    from app.auth.routes import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # 注册检测相关蓝图
    try:
        from app.detection.routes import detection_bp
        app.register_blueprint(detection_bp)
        print("✅ 检测蓝图注册成功")
    except Exception as e:
        print(f"❌ 检测蓝图注册失败: {str(e)}")

    # 注册前端兼容蓝图
    try:
        from app.detect_compat import detect_compat_bp
        app.register_blueprint(detect_compat_bp)
        print("✅ 前端兼容蓝图注册成功 - 支持 /detect/detect 路由")
    except Exception as e:
        print(f"❌ 前端兼容蓝图注册失败: {str(e)}")

    return appfrom flask import Flask
from flask_migrate import Migrate
from flask_cors import CORS
from config import config
from app.utils.redis_helper import init_redis
from app.utils.mongodb_helper import mongodb
from app.models import db
import os
import sys

migrate = Migrate()

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # 创建必要的目录
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'app/static/uploads'), exist_ok=True)
    os.makedirs(app.config.get('RESULTS_FOLDER', 'app/static/results'), exist_ok=True)
    os.makedirs(app.config.get('YOLO_WEIGHTS_PATH', 'weights'), exist_ok=True)

    # 调试：检查文件和路径
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"📁 Python路径: {sys.path[:3]}...")
    print(f"📄 yolo_service.py存在: {os.path.exists('app/services/yolo_service.py')}")
    print(f"📄 yolov5目录存在: {os.path.exists('yolov5')}")
    print(f"📄 weights目录存在: {os.path.exists('weights')}")

    # 初始化数据库
    db.init_app(app)

    # 初始化数据库迁移
    migrate.init_app(app, db)

    # 初始化CORS
    CORS(app,
         origins=app.config.get('CORS_ORIGINS', ['*']),
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         supports_credentials=True)

    # 初始化Redis
    init_redis(app)

    # 初始化MongoDB
    mongodb.init_app(app)

    # 初始化YOLO服务
    try:
        from app.detection.routes import init_yolo_service
        init_yolo_service(app)
    except Exception as e:
        print(f"❌ YOLO服务初始化失败: {str(e)}")

    # 注册蓝图
    from app.auth.routes import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    try:
        from app.detection.routes import detection_bp
        app.register_blueprint(detection_bp)
    except Exception as e:
        print(f"❌ 检测蓝图注册失败: {str(e)}")

    return app