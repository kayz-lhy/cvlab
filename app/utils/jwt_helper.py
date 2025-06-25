import jwt
from datetime import datetime, timedelta
from flask import current_app
from app.utils.redis_helper import redis_client

class JWTHelper:
    @staticmethod
    def generate_token(user_id):
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')

        # 将token存储到Redis中，过期时间24小时
        redis_client.setex(f"token:{user_id}", 86400, token)

        return token

    @staticmethod
    def verify_token(token):
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload['user_id']

            # 检查token是否在Redis中存在
            stored_token = redis_client.get(f"token:{user_id}")
            if not stored_token or stored_token.decode() != token:
                return None

            return user_id
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    @staticmethod
    def revoke_token(user_id):
        # 从Redis中删除token
        redis_client.delete(f"token:{user_id}")

    @staticmethod
    def refresh_token(user_id):
        # 撤销旧token并生成新token
        JWTHelper.revoke_token(user_id)
        return JWTHelper.generate_token(user_id)