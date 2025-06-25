from pymongo import MongoClient
from flask import current_app
from datetime import datetime
import uuid

class MongoDBHelper:
    def __init__(self):
        self.client = None
        self.db = None

    def init_app(self, app):
        """初始化MongoDB连接"""
        try:
            self.client = MongoClient(app.config['MONGO_URI'])
            self.db = self.client.get_default_database()
            # 测试连接
            self.client.admin.command('ping')
            print("MongoDB连接成功")
        except Exception as e:
            print(f"MongoDB连接失败: {str(e)}")
            self.client = None
            self.db = None

    def save_detection_record(self, user_id, detection_data):
        """保存检测记录到MongoDB"""
        if not self.db:
            return None

        try:
            record = {
                'record_id': str(uuid.uuid4()),
                'user_id': user_id,
                'timestamp': datetime.utcnow(),
                'original_image_path': detection_data.get('original_image_path'),
                'result_image_path': detection_data.get('result_image_path'),
                'detections': detection_data.get('detections', []),
                'detection_count': detection_data.get('detection_count', 0),
                'model_name': detection_data.get('model_name'),
                'confidence_threshold': detection_data.get('confidence_threshold'),
                'processing_time': detection_data.get('processing_time'),
                'image_size': detection_data.get('image_size'),
                'metadata': {
                    'filename': detection_data.get('filename'),
                    'file_size': detection_data.get('file_size'),
                    'created_at': datetime.utcnow().isoformat()
                }
            }

            result = self.db.detection_records.insert_one(record)
            return str(result.inserted_id)
        except Exception as e:
            print(f"保存检测记录失败: {str(e)}")
            return None

    def get_user_detections(self, user_id, limit=50, skip=0):
        """获取用户的检测历史"""
        if not self.db:
            return []

        try:
            cursor = self.db.detection_records.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).skip(skip).limit(limit)

            records = []
            for record in cursor:
                record['_id'] = str(record['_id'])
                records.append(record)

            return records
        except Exception as e:
            print(f"获取用户检测历史失败: {str(e)}")
            return []

# 全局MongoDB实例
mongodb = MongoDBHelper()