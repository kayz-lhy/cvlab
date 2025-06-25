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
         allow_headers=[
             'Content-Type',
             'Authorization',
             'X-Requested-With',
             'Accept',
             'Origin'
         ],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         supports_credentials=True,
         max_age=3600,  # OPTIONS预检请求缓存1小时
         send_wildcard=False,  # 提高安全性
         vary_header=True  # 添加Vary头
         )

    # 初始化Redis
    init_redis(app)

    # 初始化MongoDB
    mongodb.init_app(app)

    # 初始化增强的YOLO服务（支持YOLOv5/v8切换和MongoDB权重管理）
    try:
        from app.services.yolo_service import YOLOServiceV2
        yolo_service = YOLOServiceV2(
            yolo_repo_path=app.config['YOLO_REPO_PATH'],
            weights_path=app.config['YOLO_WEIGHTS_PATH'],
            mongodb_helper=mongodb
        )
        # 将YOLO服务存储在app实例中
        app.yolo_service = yolo_service
        print("✅ 增强YOLO服务初始化成功")

        # 尝试加载默认模型（如果MongoDB中有权重文件）
        try:
            weights_list = yolo_service.list_available_weights()
            if weights_list:
                # 尝试加载第一个可用的权重
                default_weight = weights_list[0]
                success, message = yolo_service.load_model_from_mongodb(default_weight['weight_id'])
                if success:
                    print(f"✅ 自动加载默认模型: {default_weight['model_name']} ({default_weight['model_type']})")
                else:
                    print(f"⚠️ 自动加载默认模型失败: {message}")
            else:
                print("📝 MongoDB中暂无权重文件，请先上传模型权重")
        except Exception as auto_load_error:
            print(f"⚠️ 自动加载模型时出错: {str(auto_load_error)}")

    except Exception as e:
        print(f"❌ YOLO服务初始化失败: {str(e)}")
        app.yolo_service = None

    # 注册蓝图
    from app.auth.routes import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # 注册增强的检测相关蓝图
    try:
        from app.detection.routes import detection_bp
        app.register_blueprint(detection_bp)
        print("✅ 增强检测蓝图注册成功")
    except Exception as e:
        print(f"❌ 检测蓝图注册失败: {str(e)}")

    # 注册兼容性蓝图（保持前端兼容性）
    try:
        from app.detect_compat import detect_compat_bp
        app.register_blueprint(detect_compat_bp)
        print("✅ 兼容性蓝图注册成功")
    except Exception as e:
        print(f"❌ 兼容性蓝图注册失败: {str(e)}")

    return app