from flask import Flask
from flask_migrate import Migrate
from flask_cors import CORS
from config import config
from app.utils.redis_helper import init_redis
from app.utils.mongodb_helper import mongodb
from app.models import db

migrate = Migrate()

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # 创建必要的目录
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['YOLO_WEIGHTS_PATH'], exist_ok=True)

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
    from app.detection.routes import init_yolo_service
    init_yolo_service(app)

    # 注册蓝图
    from app.auth.routes import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    from app.detection.routes import detection_bp
    app.register_blueprint(detection_bp)

    migrate.init_app(app, db)

    # 初始化CORS
    CORS(app,
         origins=app.config.get('CORS_ORIGINS', ['*']),
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         supports_credentials=True)

    # 初始化Redis
    init_redis(app)

    # 注册蓝图
    from app.auth.routes import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app