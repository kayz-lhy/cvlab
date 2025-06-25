import redis
from flask import current_app

# Redis连接实例
redis_client = None

def init_redis(app):
    global redis_client
    redis_client = redis.Redis(
        host=app.config.get('REDIS_HOST', 'server'),
        port=app.config.get('REDIS_PORT', 6379),
        db=app.config.get('REDIS_DB', 0),
        decode_responses=True
    )