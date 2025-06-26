import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://postgre:987654@101.43.142.153:5432/cvlab'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Redis配置 - 默认使用localhost
    REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'
    REDIS_PORT = int(os.environ.get('REDIS_PORT') or 6379)
    REDIS_DB = int(os.environ.get('REDIS_DB') or 0)
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD') or None  # Redis密码（可选）

    # MongoDB配置
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://admin:admin@localhost:27017/cvlab?authSource=admin'

    # JWT配置
    JWT_SECRET_KEY = SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    # CORS配置
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # YOLOv5配置
    YOLO_REPO_PATH = os.path.join(os.getcwd(), 'yolov5')
    YOLO_WEIGHTS_PATH = os.path.join(os.getcwd(), 'weights')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app/static/uploads')
    RESULTS_FOLDER = os.path.join(os.getcwd(), 'app/static/results')


class DevelopmentConfig(Config):
    DEBUG = True
    # 开发环境允许所有域名
    CORS_ORIGINS = ["*"]
    # 开发环境使用localhost而不是server
    # 开发环境使用localhost
    REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://admin:admin@localhost:27017/cvlab?authSource=admin'


class ProductionConfig(Config):
    DEBUG = False
    # 生产环境只允许指定域名
    CORS_ORIGINS = [
        "https://yourdomain.com",
        "https://www.yourdomain.com",
        "*"
    ]
    # 生产环境使用server
    REDIS_HOST = 'localhost'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://admin:admin@localhost:27017/cvlab?authSource=admin'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
