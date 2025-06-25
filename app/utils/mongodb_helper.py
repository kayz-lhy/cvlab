from pymongo import MongoClient
from gridfs import GridFS
from flask import current_app
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional
from bson import Binary

class MongoDBHelper:
    def __init__(self):
        self.client = None
        self.db = None
        self.fs = None  # GridFS实例

    def init_app(self, app):
        """初始化MongoDB连接"""
        try:
            self.client = MongoClient(app.config['MONGO_URI'])
            self.db = self.client.get_default_database()
            # 初始化GridFS
            self.fs = GridFS(self.db)
            # 测试连接
            self.client.admin.command('ping')
            print("MongoDB和GridFS连接成功")

            # 创建索引
            self._create_indexes()

        except Exception as e:
            print(f"MongoDB连接失败: {str(e)}")
            self.client = None
            self.db = None
            self.fs = None

    def _create_indexes(self):
        """创建必要的索引"""
        try:
            # 为权重文件集合创建索引
            if self.db is not None:
                # 为权重ID创建唯一索引
                self.db.weight_metadata.create_index("weight_id", unique=True)
                # 为文件哈希创建唯一索引（防止重复上传）
                self.db.weight_metadata.create_index("file_hash", unique=True)
                # 为模型类型创建索引
                self.db.weight_metadata.create_index("model_type")
                # 为活跃状态创建索引
                self.db.weight_metadata.create_index("is_active")
                # 为上传时间创建索引
                self.db.weight_metadata.create_index("upload_time")
                # 为GridFS文件ID创建索引
                self.db.weight_metadata.create_index("gridfs_file_id")

                print("MongoDB索引创建成功")
        except Exception as e:
            print(f"创建MongoDB索引失败: {str(e)}")

    def save_detection_record(self, user_id, detection_data):
        """保存检测记录到MongoDB"""
        if self.db is None:
            print("❌ MongoDB数据库未连接")
            return None

        try:
            # 确保必要字段存在
            record = {
                'record_id': str(uuid.uuid4()),
                'user_id': user_id,
                'timestamp': datetime.utcnow(),
                'original_image_path': detection_data.get('original_image_path'),
                'result_image_path': detection_data.get('result_image_path'),
                'detections': detection_data.get('detections', []),
                'detection_count': detection_data.get('detection_count', 0),
                'model_name': detection_data.get('model_name', 'unknown'),
                'model_type': detection_data.get('model_type', 'unknown'),
                'weight_id': detection_data.get('weight_id'),
                'confidence_threshold': detection_data.get('confidence_threshold', 0.25),
                'processing_time': detection_data.get('processing_time', 0.0),
                'image_size': detection_data.get('image_size', []),
                'metadata': {
                    'filename': detection_data.get('filename', 'unknown.jpg'),
                    'file_size': detection_data.get('file_size', 0),
                    'created_at': datetime.utcnow().isoformat()
                }
            }

            print(f"📝 准备保存检测记录:")
            print(f"   用户ID: {user_id}")
            print(f"   检测对象数: {record['detection_count']}")
            print(f"   模型: {record['model_name']} ({record['model_type']})")
            print(f"   文件名: {record['metadata']['filename']}")

            result = self.db.detection_records.insert_one(record)
            record_id = str(result.inserted_id)

            print(f"✅ 检测记录保存成功，ID: {record_id}")
            return record_id

        except Exception as e:
            print(f"❌ 保存检测记录失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_weight_file(self, weight_record: Dict[str, Any]) -> Optional[str]:
        """保存权重文件到GridFS"""
        if self.fs is None or self.db is None:
            return None

        try:
            # 提取二进制数据
            weights_data = weight_record.pop('weights_data')

            # 将权重文件存储到GridFS
            gridfs_file_id = self.fs.put(
                weights_data,
                filename=f"{weight_record['model_name']}_{weight_record['weight_id']}.pt",
                metadata={
                    'weight_id': weight_record['weight_id'],
                    'model_name': weight_record['model_name'],
                    'model_type': weight_record['model_type'],
                    'upload_time': weight_record['upload_time'],
                    'content_type': 'application/octet-stream'
                }
            )

            # 在权重元数据中记录GridFS文件ID
            weight_record['gridfs_file_id'] = gridfs_file_id

            # 保存元数据到单独的集合
            result = self.db.weight_metadata.insert_one(weight_record)
            return str(result.inserted_id)

        except Exception as e:
            print(f"保存权重文件失败: {str(e)}")
            return None

    def get_weight_by_id(self, weight_id: str) -> Optional[Dict[str, Any]]:
        """根据权重ID获取权重记录"""
        if self.db is None or self.fs is None:
            return None

        try:
            # 获取元数据
            record = self.db.weight_metadata.find_one(
                {'weight_id': weight_id, 'is_active': True}
            )

            if not record:
                return None

            # 从GridFS获取文件数据
            try:
                gridfs_file = self.fs.get(record['gridfs_file_id'])
                record['weights_data'] = gridfs_file.read()
                gridfs_file.close()
            except Exception as e:
                print(f"从GridFS获取文件失败: {str(e)}")
                return None

            record['_id'] = str(record['_id'])
            return record

        except Exception as e:
            print(f"获取权重记录失败: {str(e)}")
            return None

    def get_weight_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """根据文件哈希获取权重记录（用于检查重复）"""
        if self.db is None:
            return None

        try:
            record = self.db.weight_metadata.find_one(
                {'file_hash': file_hash, 'is_active': True},
                {'gridfs_file_id': 0}  # 不返回GridFS文件ID
            )
            if record:
                record['_id'] = str(record['_id'])
            return record
        except Exception as e:
            print(f"根据哈希获取权重记录失败: {str(e)}")
            return None

    def list_weight_files(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出权重文件"""
        if self.db is None:
            return []

        try:
            query = {'is_active': True}
            if model_type:
                query['model_type'] = model_type

            cursor = self.db.weight_metadata.find(
                query,
                {'weights_data': 0}  # 不返回二进制数据，提高性能
            ).sort('upload_time', -1)

            records = []
            for record in cursor:
                record['_id'] = str(record['_id'])

                # 确保class_count字段存在并有正确的值
                model_info = record.get('model_info', {})
                class_count = model_info.get('class_count', 0)

                # 如果class_count为0，设置默认值80（COCO数据集）
                if class_count == 0:
                    print(f"⚠️ 权重 {record.get('model_name', 'unknown')} 的class_count为0，设置为默认值80")
                    class_count = 80
                    # 同时更新数据库记录
                    try:
                        self.db.weight_metadata.update_one(
                            {'weight_id': record['weight_id']},
                            {
                                '$set': {
                                    'model_info.class_count': 80,
                                    'updated_at': datetime.utcnow()
                                }
                            }
                        )
                    except Exception as e:
                        print(f"更新权重class_count失败: {str(e)}")

                # 直接在record中设置class_count字段，以便前端可以访问
                record['class_count'] = class_count

                print(f"📝 权重 {record.get('model_name', 'unknown')} class_count: {class_count}")
                records.append(record)

            return records
        except Exception as e:
            print(f"❌ 列出权重文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def delete_weight_file(self, weight_id: str) -> bool:
        """删除权重文件（软删除）"""
        if self.db is None or self.fs is None:
            return False

        try:
            # 获取权重记录
            record = self.db.weight_metadata.find_one({'weight_id': weight_id})
            if not record:
                return False

            # 从GridFS删除文件
            try:
                self.fs.delete(record['gridfs_file_id'])
            except Exception as e:
                print(f"从GridFS删除文件失败: {str(e)}")
                # 即使GridFS删除失败，也继续软删除元数据

            # 软删除元数据
            result = self.db.weight_metadata.update_one(
                {'weight_id': weight_id},
                {
                    '$set': {
                        'is_active': False,
                        'deleted_at': datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0

        except Exception as e:
            print(f"删除权重文件失败: {str(e)}")
            return False

    def update_weight_info(self, weight_id: str, updates: Dict[str, Any]) -> bool:
        """更新权重文件信息"""
        if self.db is None:
            return False

        try:
            updates['updated_at'] = datetime.utcnow()
            result = self.db.weight_metadata.update_one(
                {'weight_id': weight_id, 'is_active': True},
                {'$set': updates}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"更新权重文件信息失败: {str(e)}")
            return False

    def get_weight_statistics(self) -> Dict[str, Any]:
        """获取权重文件统计信息"""
        if self.db is None:
            return {}

        try:
            stats = {
                'total_weights': 0,
                'total_size': 0,
                'model_type_counts': {},
                'recent_uploads': 0
            }

            # 总数和总大小
            pipeline = [
                {'$match': {'is_active': True}},
                {'$group': {
                    '_id': None,
                    'total_count': {'$sum': 1},
                    'total_size': {'$sum': '$file_size'}
                }}
            ]
            result = list(self.db.weight_metadata.aggregate(pipeline))
            if result:
                stats['total_weights'] = result[0]['total_count']
                stats['total_size'] = result[0]['total_size']

            # 按模型类型分组统计
            pipeline = [
                {'$match': {'is_active': True}},
                {'$group': {
                    '_id': '$model_type',
                    'count': {'$sum': 1}
                }}
            ]
            for item in self.db.weight_metadata.aggregate(pipeline):
                stats['model_type_counts'][item['_id']] = item['count']

            # 最近7天上传的权重数量
            week_ago = datetime.utcnow() - timedelta(days=7)
            stats['recent_uploads'] = self.db.weight_metadata.count_documents({
                'is_active': True,
                'upload_time': {'$gte': week_ago}
            })

            return stats
        except Exception as e:
            print(f"获取权重文件统计失败: {str(e)}")
            return {}

    def get_user_detections(self, user_id, limit=50, skip=0):
        """获取用户的检测历史"""
        if self.db is None:
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

    def get_detection_statistics(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """获取检测统计信息"""
        if self.db is None:
            print("❌ MongoDB数据库未连接")
            return {}

        try:
            match_condition = {}
            if user_id:
                match_condition['user_id'] = user_id

            print(f"📊 开始统计，筛选条件: {match_condition}")

            stats = {
                'total_detections': 0,
                'total_objects_detected': 0,
                'model_usage': {},
                'recent_detections': 0
            }

            # 检查是否有数据
            total_records = self.db.detection_records.count_documents(match_condition)
            print(f"📊 找到 {total_records} 条记录")

            if total_records == 0:
                print("📊 没有找到检测记录，返回空统计")
                return stats

            # 总检测次数和检测到的对象总数
            pipeline = [
                {'$match': match_condition},
                {'$group': {
                    '_id': None,
                    'total_detections': {'$sum': 1},
                    'total_objects': {'$sum': '$detection_count'}
                }}
            ]

            print(f"📊 执行聚合查询: {pipeline}")
            result = list(self.db.detection_records.aggregate(pipeline))
            print(f"📊 聚合结果: {result}")

            if result:
                stats['total_detections'] = result[0]['total_detections']
                stats['total_objects_detected'] = result[0]['total_objects']

            # 按模型使用情况统计
            pipeline_models = [
                {'$match': match_condition},
                {'$group': {
                    '_id': {
                        'model_name': '$model_name',
                        'model_type': '$model_type'
                    },
                    'count': {'$sum': 1},
                    'avg_processing_time': {'$avg': '$processing_time'}
                }}
            ]

            print(f"📊 执行模型统计查询: {pipeline_models}")
            models_result = list(self.db.detection_records.aggregate(pipeline_models))
            print(f"📊 模型统计结果: {models_result}")

            for item in models_result:
                model_name = item['_id'].get('model_name')
                model_type = item['_id'].get('model_type')

                if model_name and model_type:
                    model_key = f"{model_name} ({model_type})"
                    stats['model_usage'][model_key] = {
                        'count': item['count'],
                        'avg_processing_time': round(item['avg_processing_time'], 3) if item['avg_processing_time'] else 0
                    }

            # 最近24小时的检测次数
            day_ago = datetime.utcnow() - timedelta(hours=24)
            recent_match = match_condition.copy()
            recent_match['timestamp'] = {'$gte': day_ago}

            stats['recent_detections'] = self.db.detection_records.count_documents(recent_match)
            print(f"📊 最近24小时检测次数: {stats['recent_detections']}")

            print(f"📊 最终统计结果: {stats}")
            return stats

        except Exception as e:
            print(f"❌ 获取检测统计失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'total_detections': 0,
                'total_objects_detected': 0,
                'model_usage': {},
                'recent_detections': 0
            }

    def cleanup_orphaned_gridfs_files(self):
        """清理孤立的GridFS文件"""
        if self.db is None or self.fs is None:
            return False

        try:
            # 获取所有活跃权重的GridFS文件ID
            active_file_ids = set()
            for record in self.db.weight_metadata.find({'is_active': True}, {'gridfs_file_id': 1}):
                active_file_ids.add(record['gridfs_file_id'])

            # 获取所有GridFS文件
            deleted_count = 0
            for grid_file in self.fs.find():
                if grid_file._id not in active_file_ids:
                    # 删除孤立的文件
                    self.fs.delete(grid_file._id)
                    deleted_count += 1

            print(f"清理了 {deleted_count} 个孤立的GridFS文件")
            return True

        except Exception as e:
            print(f"清理孤立文件失败: {str(e)}")
            return False

# 全局MongoDB实例
mongodb = MongoDBHelper()