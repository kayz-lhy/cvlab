import redis
from flask import current_app

# Redis连接实例
redis_client = None

def init_redis(app):
    """初始化Redis连接"""
    global redis_client

    try:
        redis_config = {
            'host': app.config.get('REDIS_HOST', 'localhost'),
            'port': app.config.get('REDIS_PORT', 6379),
            'db': app.config.get('REDIS_DB', 0),
            'decode_responses': True,
            'socket_timeout': 5,  # 添加超时设置
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30  # 健康检查间隔
        }

        # 如果配置了密码，添加密码
        redis_password = app.config.get('REDIS_PASSWORD')
        if redis_password:
            redis_config['password'] = redis_password

        redis_client = redis.Redis(**redis_config)

        # 测试连接
        redis_client.ping()
        print(f"✅ Redis连接成功: {app.config.get('REDIS_HOST', 'localhost')}:{app.config.get('REDIS_PORT', 6379)}")

    except redis.ConnectionError as e:
        print(f"❌ Redis连接失败: {str(e)}")
        print(f"📝 尝试连接: {app.config.get('REDIS_HOST', 'localhost')}:{app.config.get('REDIS_PORT', 6379)}")
        redis_client = None
    except Exception as e:
        print(f"❌ Redis初始化失败: {str(e)}")
        redis_client = None

def get_redis_client():
    """获取Redis客户端，如果连接断开则尝试重连"""
    global redis_client

    if redis_client is None:
        return None

    try:
        # 测试连接是否有效
        redis_client.ping()
        return redis_client
    except (redis.ConnectionError, redis.TimeoutError):
        print("⚠️ Redis连接断开，尝试重连...")
        try:
            redis_client.connection_pool.disconnect()
            redis_client.ping()
            print("✅ Redis重连成功")
            return redis_client
        except Exception as e:
            print(f"❌ Redis重连失败: {str(e)}")
            return None

def safe_redis_operation(operation, *args, **kwargs):
    """安全的Redis操作，包含错误处理"""
    client = get_redis_client()
    if client is None:
        print("⚠️ Redis客户端不可用，跳过操作")
        return None

    try:
        return getattr(client, operation)(*args, **kwargs)
    except Exception as e:
        print(f"❌ Redis操作 {operation} 失败: {str(e)}")
        return None