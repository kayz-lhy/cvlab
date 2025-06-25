import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import uuid
from datetime import datetime
import time
import base64
import shutil
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import hashlib

class ModelType(Enum):
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"

class YOLOServiceV2:
    def __init__(self, yolo_repo_path, weights_path, mongodb_helper):
        self.yolo_repo_path = yolo_repo_path
        self.weights_path = weights_path
        self.mongodb = mongodb_helper
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_model_type = None
        self.current_weights_info = None

        # 将YOLOv5仓库路径添加到系统路径
        if yolo_repo_path not in sys.path:
            sys.path.insert(0, yolo_repo_path)

    def _detect_model_type(self, weights_file_path: str) -> ModelType:
        """通过分析权重文件检测模型类型"""
        try:
            # 尝试加载权重文件查看结构
            checkpoint = torch.load(weights_file_path, map_location='cpu')

            # YOLOv8的特征检测
            if 'model' in checkpoint and hasattr(checkpoint.get('model'), 'yaml'):
                yaml_content = str(checkpoint['model'].yaml)
                if 'yolov8' in yaml_content.lower() or 'v8' in yaml_content.lower():
                    return ModelType.YOLOV8

            # 检查模型架构信息
            if 'model' in checkpoint:
                model_info = str(checkpoint['model'])
                if 'C2f' in model_info or 'SPPF' in model_info:  # YOLOv8特有的模块
                    return ModelType.YOLOV8
                elif 'C3' in model_info or 'SPP' in model_info:  # YOLOv5特有的模块
                    return ModelType.YOLOV5

            # 默认假设为YOLOv5（向后兼容）
            return ModelType.YOLOV5

        except Exception as e:
            print(f"检测模型类型失败: {str(e)}")
            return ModelType.YOLOV5

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"计算文件哈希失败: {str(e)}")
            return ""

    def upload_weights_to_mongodb(self, weights_file_path: str, model_name: str,
                                  description: str = "", model_type: Optional[ModelType] = None) -> Dict[str, Any]:
        """上传权重文件到MongoDB"""
        try:
            if not os.path.exists(weights_file_path):
                return {'success': False, 'error': '权重文件不存在'}

            # 自动检测模型类型（如果未指定）
            if model_type is None:
                model_type = self._detect_model_type(weights_file_path)

            # 读取权重文件
            with open(weights_file_path, 'rb') as f:
                weights_data = f.read()

            # 计算文件哈希
            file_hash = self._calculate_file_hash(weights_file_path)

            # 检查是否已存在相同哈希的权重
            existing_weight = self.mongodb.get_weight_by_hash(file_hash)
            if existing_weight:
                return {
                    'success': False,
                    'error': f'相同的权重文件已存在: {existing_weight["model_name"]}'
                }

            # 获取文件信息
            file_size = len(weights_data)
            file_stats = os.stat(weights_file_path)

            # 尝试加载权重获取模型信息
            model_info = self._extract_model_info(weights_file_path)

            # 构建权重记录
            weight_record = {
                'weight_id': str(uuid.uuid4()),
                'model_name': model_name,
                'model_type': model_type.value,
                'description': description,
                'file_hash': file_hash,
                'file_size': file_size,
                'weights_data': weights_data,  # 二进制数据
                'model_info': model_info,
                'upload_time': datetime.utcnow(),
                'created_by': 'system',  # 可以从当前用户获取
                'is_active': True,
                'metadata': {
                    'original_filename': os.path.basename(weights_file_path),
                    'upload_timestamp': datetime.utcnow().isoformat()
                }
            }

            # 保存到MongoDB
            result_id = self.mongodb.save_weight_file(weight_record)

            if result_id:
                return {
                    'success': True,
                    'weight_id': weight_record['weight_id'],
                    'model_type': model_type.value,
                    'file_size': file_size,
                    'message': f'权重文件上传成功: {model_name}'
                }
            else:
                return {'success': False, 'error': '保存到数据库失败'}

        except Exception as e:
            return {'success': False, 'error': f'上传权重文件失败: {str(e)}'}

    def _extract_model_info(self, weights_file_path: str) -> Dict[str, Any]:
        """提取模型信息 - 改进版本"""
        try:
            checkpoint = torch.load(weights_file_path, map_location='cpu')

            info = {
                'classes': {},
                'class_count': 0,
                'input_size': None,
                'architecture': 'unknown'
            }

            # 提取类别信息 - 多种方式尝试（使用独立的if而不是elif）
            classes_dict = None

            print(f"🔍 开始分析权重文件结构...")
            print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")

            # 方法1: 直接从checkpoint获取names
            if 'names' in checkpoint and checkpoint['names']:
                classes_dict = checkpoint['names']
                print(f"✅ 方法1成功：从checkpoint['names']获取到 {len(classes_dict)} 个类别")

            # 方法2: 从model获取names
            if not classes_dict and 'model' in checkpoint:
                model = checkpoint['model']
                print(f"🔍 Model type: {type(model)}")
                print(f"🔍 Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}...")

                # 2a: 直接从model.names获取
                if hasattr(model, 'names') and model.names:
                    classes_dict = model.names
                    print(f"✅ 方法2a成功：从model.names获取到 {len(classes_dict)} 个类别")

                # 2b: 从model.model.names获取
                elif hasattr(model, 'model') and hasattr(model.model, 'names') and model.model.names:
                    classes_dict = model.model.names
                    print(f"✅ 方法2b成功：从model.model.names获取到 {len(classes_dict)} 个类别")

                # 2c: 从model[-1].names获取（某些YOLOv5版本）
                elif hasattr(model, '__getitem__'):
                    try:
                        last_layer = model[-1]
                        if hasattr(last_layer, 'names') and last_layer.names:
                            classes_dict = last_layer.names
                            print(f"✅ 方法2c成功：从model[-1].names获取到 {len(classes_dict)} 个类别")
                    except:
                        pass

            # 方法3: 从训练参数中获取
            if not classes_dict and 'train_args' in checkpoint:
                train_args = checkpoint['train_args']
                if hasattr(train_args, 'data') and train_args.data:
                    try:
                        # 尝试读取数据配置文件
                        data_path = train_args.data
                        if isinstance(data_path, str) and data_path.endswith('.yaml'):
                            classes_dict = self._extract_classes_from_yaml(data_path)
                            if classes_dict:
                                print(f"✅ 方法3成功：从yaml文件获取到 {len(classes_dict)} 个类别")
                    except Exception as e:
                        print(f"⚠️ 从yaml文件提取类别失败: {str(e)}")

            # 方法4: 从模型结构推断（针对标准数据集）
            if not classes_dict:
                classes_dict = self._infer_classes_from_model_structure(checkpoint)
                if classes_dict:
                    print(f"✅ 方法4成功：从模型结构推断到 {len(classes_dict)} 个类别")

            # 方法5: 使用默认COCO类别
            if not classes_dict:
                print("⚠️ 所有方法都失败，使用默认COCO 80类别")
                classes_dict = self._get_default_coco_classes()

            # 设置类别信息
            if classes_dict:
                # 确保类别字典的格式正确（id: name）
                if isinstance(classes_dict, dict):
                    # 如果键不是整数，尝试转换
                    if not all(isinstance(k, int) for k in classes_dict.keys()):
                        try:
                            classes_dict = {int(k): v for k, v in classes_dict.items()}
                        except:
                            # 如果转换失败，重新编号
                            classes_dict = {i: v for i, v in enumerate(classes_dict.values())}
                elif isinstance(classes_dict, list):
                    # 如果是列表，转换为字典
                    classes_dict = {i: name for i, name in enumerate(classes_dict)}

                info['classes'] = classes_dict
                info['class_count'] = len(classes_dict)
                print(f"✅ 最终设置类别数量: {info['class_count']}")

            # 提取其他信息
            if 'model' in checkpoint:
                model = checkpoint['model']

                # 获取模型架构信息
                if hasattr(model, 'yaml'):
                    try:
                        yaml_info = model.yaml
                        if isinstance(yaml_info, dict):
                            info['architecture'] = yaml_info.get('backbone', 'unknown')
                        else:
                            info['architecture'] = str(yaml_info)[:50]  # 截取前50字符
                    except:
                        pass

                # 获取stride信息
                if hasattr(model, 'stride'):
                    try:
                        info['stride'] = model.stride.tolist() if hasattr(model.stride, 'tolist') else model.stride
                    except:
                        pass

            return info

        except Exception as e:
            print(f"❌ 提取模型信息失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 返回默认的COCO类别信息
            default_classes = self._get_default_coco_classes()
            return {
                'classes': default_classes,
                'class_count': len(default_classes),
                'extraction_error': str(e)
            }

    def _extract_classes_from_yaml(self, yaml_path: str) -> Optional[Dict[int, str]]:
        """从YAML文件提取类别信息"""
        try:
            import yaml

            # 如果是相对路径，尝试在多个位置查找
            search_paths = [
                yaml_path,
                os.path.join(os.getcwd(), yaml_path),
                os.path.join(self.yolo_repo_path, yaml_path),
                os.path.join(self.yolo_repo_path, 'data', os.path.basename(yaml_path))
            ]

            for path in search_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

                    if 'names' in data:
                        names = data['names']
                        if isinstance(names, list):
                            return {i: name for i, name in enumerate(names)}
                        elif isinstance(names, dict):
                            return names
                    break

            return None
        except Exception as e:
            print(f"从YAML文件提取类别失败: {str(e)}")
            return None

    def _infer_classes_from_model_structure(self, checkpoint: Dict) -> Optional[Dict[int, str]]:
        """从模型结构推断类别信息"""
        try:
            # 尝试从模型的输出层推断类别数量
            if 'model' in checkpoint:
                model = checkpoint['model']

                # 查找输出层的类别数量
                class_count = None

                # 方法1: 查找最后一层的输出维度
                if hasattr(model, 'model') and len(model.model) > 0:
                    try:
                        last_layer = model.model[-1]
                        if hasattr(last_layer, 'nc'):
                            class_count = last_layer.nc
                        elif hasattr(last_layer, 'anchors') and hasattr(last_layer, 'no'):
                            # YOLOv5的Detect层
                            class_count = last_layer.no - 5  # no = nc + 5 (x,y,w,h,conf)
                    except:
                        pass

                # 方法2: 从state_dict推断
                if not class_count and hasattr(model, 'state_dict'):
                    try:
                        state_dict = model.state_dict()
                        for key, tensor in state_dict.items():
                            if 'head' in key.lower() or 'classifier' in key.lower():
                                if tensor.dim() >= 2:
                                    class_count = tensor.shape[0]
                                    break
                    except:
                        pass

                # 根据推断的类别数量返回对应的标准数据集类别
                if class_count:
                    if class_count == 80:
                        return self._get_default_coco_classes()
                    elif class_count == 1:
                        return {0: 'object'}
                    elif class_count <= 20:
                        # 可能是自定义的小数据集
                        return {i: f'class_{i}' for i in range(class_count)}

            return None
        except Exception as e:
            print(f"从模型结构推断类别失败: {str(e)}")
            return None

    def _get_default_coco_classes(self) -> Dict[int, str]:
        """获取默认的COCO数据集类别"""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
            15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }

    def load_model_from_mongodb(self, weight_id: str) -> Tuple[bool, str]:
        """从MongoDB加载模型"""
        try:
            # 从MongoDB获取权重记录
            weight_record = self.mongodb.get_weight_by_id(weight_id)
            if not weight_record:
                return False, "权重记录不存在"

            # 检查模型类型
            model_type = ModelType(weight_record['model_type'])

            # 将权重数据写入临时文件
            temp_weights_path = os.path.join(self.weights_path, f"temp_{weight_id}.pt")
            os.makedirs(os.path.dirname(temp_weights_path), exist_ok=True)

            with open(temp_weights_path, 'wb') as f:
                f.write(weight_record['weights_data'])

            # 根据模型类型加载
            success, message = self._load_model_by_type(temp_weights_path, model_type)

            if success:
                self.current_model_type = model_type
                self.current_weights_info = {
                    'weight_id': weight_id,
                    'model_name': weight_record['model_name'],
                    'model_type': model_type.value,
                    'description': weight_record.get('description', ''),
                    'classes': weight_record.get('model_info', {}).get('classes', {}),
                    'class_count': weight_record.get('model_info', {}).get('class_count', 0)
                }

                # 如果MongoDB中的类别信息为空或错误，尝试从实际加载的模型中获取
                if self.current_weights_info['class_count'] == 0:
                    print("⚠️ MongoDB中类别信息为空，尝试从加载的模型获取...")
                    model_classes = self._get_classes_from_loaded_model()
                    if model_classes:
                        self.current_weights_info['classes'] = model_classes
                        self.current_weights_info['class_count'] = len(model_classes)

                        # 更新MongoDB中的记录
                        self._update_mongodb_class_info(weight_id, model_classes)

            # 清理临时文件
            try:
                os.remove(temp_weights_path)
            except:
                pass

            return success, message

        except Exception as e:
            return False, f"从MongoDB加载模型失败: {str(e)}"

    def _get_classes_from_loaded_model(self) -> Optional[Dict[int, str]]:
        """从已加载的模型获取类别信息"""
        try:
            if not self.model:
                return None

            model_classes = None

            if self.current_model_type == ModelType.YOLOV8:
                # YOLOv8模型
                if hasattr(self.model, 'names'):
                    model_classes = self.model.names
            else:
                # YOLOv5模型
                if hasattr(self.model, 'names'):
                    model_classes = self.model.names
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                    model_classes = self.model.model.names

            # 格式化类别信息
            if model_classes:
                if isinstance(model_classes, list):
                    return {i: name for i, name in enumerate(model_classes)}
                elif isinstance(model_classes, dict):
                    # 确保键是整数
                    formatted_classes = {}
                    for k, v in model_classes.items():
                        try:
                            formatted_classes[int(k)] = str(v)
                        except:
                            continue
                    return formatted_classes if formatted_classes else None

            return None

        except Exception as e:
            print(f"从加载的模型获取类别信息失败: {str(e)}")
            return None

    def _update_mongodb_class_info(self, weight_id: str, classes: Dict[int, str]):
        """更新MongoDB中的类别信息"""
        try:
            update_data = {
                'model_info.classes': classes,
                'model_info.class_count': len(classes),
                'model_info.updated_at': datetime.utcnow()
            }

            success = self.mongodb.update_weight_info(weight_id, update_data)
            if success:
                print(f"✅ 已更新MongoDB中权重 {weight_id} 的类别信息: {len(classes)} 个类别")
            else:
                print(f"⚠️ 更新MongoDB中权重 {weight_id} 的类别信息失败")

        except Exception as e:
            print(f"更新MongoDB类别信息失败: {str(e)}")

    def _load_model_by_type(self, weights_path: str, model_type: ModelType) -> Tuple[bool, str]:
        """根据模型类型加载模型"""
        try:
            if model_type == ModelType.YOLOV8:
                return self._load_yolov8_model(weights_path)
            elif model_type == ModelType.YOLOV5:
                return self._load_yolov5_model(weights_path)
            else:
                return False, f"不支持的模型类型: {model_type.value}"
        except Exception as e:
            return False, f"加载模型失败: {str(e)}"

    def _load_yolov8_model(self, weights_path: str) -> Tuple[bool, str]:
        """加载YOLOv8模型"""
        try:
            from ultralytics import YOLO

            print(f"正在加载YOLOv8模型: {weights_path}")
            self.model = YOLO(weights_path)

            print(f"✅ YOLOv8模型加载成功, 设备: {self.device}")
            return True, "YOLOv8模型加载成功"

        except ImportError:
            return False, "Ultralytics库未安装，请安装: pip install ultralytics"
        except Exception as e:
            print(f"❌ YOLOv8模型加载失败: {str(e)}")
            return False, f"YOLOv8模型加载失败: {str(e)}"

    def _load_yolov5_model(self, weights_path: str) -> Tuple[bool, str]:
        """加载YOLOv5模型"""
        try:
            # 方法1: 使用torch.hub加载
            try:
                print(f"正在使用torch.hub加载YOLOv5模型: {weights_path}")
                self.model = torch.hub.load(
                    self.yolo_repo_path,
                    'custom',
                    path=weights_path,
                    source='local',
                    force_reload=True,
                    trust_repo=True
                )
                self.model.to(self.device)
                print(f"✅ YOLOv5 torch.hub加载成功, 设备: {self.device}")
                return True, "YOLOv5模型加载成功"

            except Exception as hub_error:
                print(f"❌ torch.hub加载失败: {str(hub_error)}")

                # 方法2: 尝试使用Ultralytics YOLO（兼容性）
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(weights_path)
                    print(f"✅ YOLOv5 Ultralytics加载成功")
                    return True, "YOLOv5模型加载成功（Ultralytics）"
                except Exception as ultralytics_error:
                    print(f"❌ Ultralytics加载失败: {str(ultralytics_error)}")
                    return False, f"YOLOv5模型加载失败: {str(hub_error)}"

        except Exception as e:
            return False, f"YOLOv5模型加载失败: {str(e)}"

    def detect_image(self, image_path: str, conf_threshold: float = 0.25,
                     save_result: bool = True) -> Dict[str, Any]:
        """检测图片中的目标（支持YOLOv5和YOLOv8）"""
        if not self.model:
            return {'success': False, 'error': '模型未加载'}

        start_time = time.time()

        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return {'success': False, 'error': '无法读取图片'}

            # 获取图片信息
            img_size = img.shape[:2]  # (height, width)
            file_size = os.path.getsize(image_path)

            # 根据模型类型进行推理
            if self.current_model_type == ModelType.YOLOV8:
                detections = self._detect_with_yolov8(image_path, conf_threshold)
                result_image_base64, result_image_path = self._generate_yolov8_result_image(
                    image_path, conf_threshold
                )
            else:  # YOLOv5
                detections = self._detect_with_yolov5(image_path, conf_threshold)
                result_image_base64, result_image_path = self._generate_yolov5_result_image(
                    image_path, conf_threshold
                )

            processing_time = round(time.time() - start_time, 3)

            return {
                'success': True,
                'detections': detections,
                'detection_count': len(detections),
                'model_name': self.current_weights_info.get('model_name', 'unknown') if self.current_weights_info else 'unknown',
                'model_type': self.current_model_type.value if self.current_model_type else 'unknown',
                'weight_id': self.current_weights_info.get('weight_id') if self.current_weights_info else None,
                'confidence_threshold': conf_threshold,
                'processing_time': processing_time,
                'image_size': img_size,
                'file_size': file_size,
                'result_image_base64': result_image_base64,
                'result_image_path': result_image_path,
                'original_image_path': image_path,
                'filename': os.path.basename(image_path)
            }

        except Exception as e:
            error_msg = f"检测过程中出错: {str(e)}"
            print(error_msg)
            return {'success': False, 'error': error_msg}

    def _detect_with_yolov8(self, image_path: str, conf_threshold: float) -> List[Dict[str, Any]]:
        """使用YOLOv8进行检测"""
        try:
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            detections = []

            # 获取类别名称
            class_names = self._get_current_class_names()

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        if conf >= conf_threshold:
                            # 获取类别名称
                            class_name = class_names.get(cls, f'class_{cls}')

                            detection = {
                                'class_id': cls,
                                'class_name': class_name,
                                'confidence': round(float(conf), 3),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            }
                            detections.append(detection)

            return detections

        except Exception as e:
            print(f"YOLOv8检测失败: {str(e)}")
            return []

    def _detect_with_yolov5(self, image_path: str, conf_threshold: float) -> List[Dict[str, Any]]:
        """使用YOLOv5进行检测"""
        try:
            results = self.model(image_path, size=640)
            results.conf = conf_threshold

            detections = []

            # 获取类别名称
            class_names = self._get_current_class_names()

            df = results.pandas().xyxy[0]

            for index, row in df.iterrows():
                if row['confidence'] >= conf_threshold:
                    cls = int(row['class'])
                    class_name = class_names.get(cls, row.get('name', f'class_{cls}'))

                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': round(float(row['confidence']), 3),
                        'bbox': [
                            int(row['xmin']),
                            int(row['ymin']),
                            int(row['xmax']),
                            int(row['ymax'])
                        ]
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            print(f"YOLOv5检测失败: {str(e)}")
            return []

    def _get_current_class_names(self) -> Dict[int, str]:
        """获取当前模型的类别名称"""
        try:
            # 优先从current_weights_info获取
            if self.current_weights_info and self.current_weights_info.get('classes'):
                return self.current_weights_info['classes']

            # 然后从实际加载的模型获取
            model_classes = self._get_classes_from_loaded_model()
            if model_classes:
                return model_classes

            # 最后使用默认COCO类别
            return self._get_default_coco_classes()

        except Exception as e:
            print(f"获取类别名称失败: {str(e)}")
            return self._get_default_coco_classes()

    def _generate_yolov8_result_image(self, image_path: str, conf_threshold: float) -> Tuple[Optional[str], Optional[str]]:
        """生成YOLOv8结果图片"""
        try:
            results = self.model(image_path, conf=conf_threshold, verbose=False)

            # 获取绘制了检测框的图片
            result_img = results[0].plot()

            # 转换为RGB格式
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image并生成base64
            pil_img = Image.fromarray(result_img_rgb)

            from io import BytesIO
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # 保存文件
            result_image_path = self._save_result_image_file(result_img, image_path)

            return img_base64, result_image_path

        except Exception as e:
            print(f"生成YOLOv8结果图片失败: {str(e)}")
            return self._get_original_image_base64(image_path), None

    def _generate_yolov5_result_image(self, image_path: str, conf_threshold: float) -> Tuple[Optional[str], Optional[str]]:
        """生成YOLOv5结果图片"""
        try:
            results = self.model(image_path, size=640)
            results.conf = conf_threshold

            result_img = results.render()[0]

            # 转换为RGB格式
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image并生成base64
            pil_img = Image.fromarray(result_img_rgb)

            from io import BytesIO
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # 保存文件
            result_image_path = self._save_result_image_file(result_img, image_path)

            return img_base64, result_image_path

        except Exception as e:
            print(f"生成YOLOv5结果图片失败: {str(e)}")
            return self._get_original_image_base64(image_path), None

    def _get_original_image_base64(self, image_path: str) -> Optional[str]:
        """获取原图像的base64编码"""
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            return base64.b64encode(img_bytes).decode('utf-8')
        except:
            return None

    def _save_result_image_file(self, result_img, original_path: str) -> Optional[str]:
        """保存结果图片文件"""
        try:
            from flask import current_app

            result_dir = current_app.config['RESULTS_FOLDER']
            os.makedirs(result_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            original_name = os.path.splitext(os.path.basename(original_path))[0]
            filename = f"result_{original_name}_{timestamp}_{unique_id}.jpg"
            result_path = os.path.join(result_dir, filename)

            cv2.imwrite(result_path, result_img)
            return os.path.relpath(result_path, 'app/static')

        except Exception as e:
            print(f"保存结果图片文件失败: {str(e)}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息 - 改进版本"""
        if not self.model:
            return {'loaded': False}

        try:
            # 首先尝试从实际加载的模型获取类别信息
            model_classes = self._get_classes_from_loaded_model()

            # 如果从模型获取到了类别信息，使用它；否则使用存储的信息
            if model_classes:
                classes_dict = model_classes
                class_source = 'live_model'
            else:
                classes_dict = self.current_weights_info.get('classes', {}) if self.current_weights_info else {}
                class_source = 'stored_info'

            # 如果还是没有类别信息，使用默认COCO类别
            if not classes_dict:
                classes_dict = self._get_default_coco_classes()
                class_source = 'default_coco'

            return {
                'loaded': True,
                'weight_id': self.current_weights_info.get('weight_id') if self.current_weights_info else None,
                'model_name': self.current_weights_info.get('model_name') if self.current_weights_info else 'unknown',
                'model_type': self.current_model_type.value if self.current_model_type else 'unknown',
                'description': self.current_weights_info.get('description', '') if self.current_weights_info else '',
                'device': str(self.device),
                'classes': classes_dict,
                'class_count': len(classes_dict),
                'class_source': class_source
            }

        except Exception as e:
            print(f"获取模型信息失败: {str(e)}")
            return {
                'loaded': False,
                'error': str(e),
                'fallback_classes': self._get_default_coco_classes(),
                'fallback_class_count': 80
            }

    def list_available_weights(self) -> List[Dict[str, Any]]:
        """列出MongoDB中可用的权重文件"""
        try:
            weights_list = self.mongodb.list_weight_files()

            # 格式化权重列表
            formatted_weights = []
            for weight in weights_list:
                # 确保class_count字段存在
                class_count = weight.get('class_count', 0)

                # 如果没有class_count，尝试从model_info中获取
                if class_count == 0:
                    model_info = weight.get('model_info', {})
                    class_count = model_info.get('class_count', 0)

                # 如果还是0，设置默认值
                if class_count == 0:
                    class_count = 80  # COCO数据集默认80类
                    print(f"⚠️ 权重 {weight.get('model_name', 'unknown')} class_count为0，使用默认值80")

                formatted_weight = {
                    'weight_id': weight['weight_id'],
                    'model_name': weight['model_name'],
                    'model_type': weight['model_type'],
                    'description': weight.get('description', ''),
                    'file_size': weight['file_size'],
                    'upload_time': weight['upload_time'].isoformat() if weight.get('upload_time') else None,
                    'class_count': class_count,
                    'is_active': weight.get('is_active', True),
                    'is_current': (self.current_weights_info and
                                   self.current_weights_info.get('weight_id') == weight['weight_id'])
                }

                print(f"📝 格式化权重 {formatted_weight['model_name']}: class_count={formatted_weight['class_count']}")
                formatted_weights.append(formatted_weight)

            return formatted_weights

        except Exception as e:
            print(f"❌ 列出权重文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def delete_weight_from_mongodb(self, weight_id: str) -> Tuple[bool, str]:
        """从MongoDB删除权重文件"""
        try:
            # 检查是否为当前使用的权重
            if (self.current_weights_info and
                    self.current_weights_info.get('weight_id') == weight_id):
                return False, "无法删除当前正在使用的权重文件"

            success = self.mongodb.delete_weight_file(weight_id)
            if success:
                return True, "权重文件删除成功"
            else:
                return False, "权重文件删除失败"

        except Exception as e:
            return False, f"删除权重文件失败: {str(e)}"

    def switch_model_type(self, target_type: ModelType) -> Tuple[bool, str]:
        """切换模型类型（在相同权重下）"""
        if not self.current_weights_info:
            return False, "当前没有加载任何模型"

        if self.current_model_type == target_type:
            return True, f"当前已经是{target_type.value}模型"

        # 重新加载当前权重，但使用新的模型类型
        weight_id = self.current_weights_info['weight_id']
        return self.load_model_from_mongodb(weight_id)

    def debug_weight_file_structure(self, weights_file_path: str) -> Dict[str, Any]:
        """调试权重文件结构的详细信息"""
        try:
            checkpoint = torch.load(weights_file_path, map_location='cpu')

            debug_info = {
                'file_path': weights_file_path,
                'file_size': os.path.getsize(weights_file_path),
                'checkpoint_keys': list(checkpoint.keys()),
                'model_analysis': {},
                'names_locations': [],
                'potential_class_info': []
            }

            print(f"🔍 调试权重文件: {weights_file_path}")
            print(f"🔍 文件大小: {debug_info['file_size'] / (1024*1024):.2f} MB")
            print(f"🔍 主要键: {debug_info['checkpoint_keys']}")

            # 分析每个主要键的内容
            for key in checkpoint.keys():
                try:
                    value = checkpoint[key]
                    key_info = {
                        'type': type(value).__name__,
                        'size': len(value) if hasattr(value, '__len__') else 'N/A'
                    }

                    # 特别检查names相关信息
                    if 'names' in key.lower():
                        key_info['content'] = value
                        debug_info['names_locations'].append({
                            'location': key,
                            'content': value,
                            'type': type(value).__name__
                        })

                    debug_info[f'key_{key}'] = key_info

                except Exception as e:
                    debug_info[f'key_{key}'] = {'error': str(e)}

            # 深度分析model键
            if 'model' in checkpoint:
                model = checkpoint['model']
                model_info = {
                    'type': type(model).__name__,
                    'attributes': [attr for attr in dir(model) if not attr.startswith('_')][:20],
                    'has_names': hasattr(model, 'names'),
                    'names_content': getattr(model, 'names', None) if hasattr(model, 'names') else None
                }

                # 检查model.model
                if hasattr(model, 'model'):
                    model_info['has_model_attr'] = True
                    model_info['model_type'] = type(model.model).__name__
                    model_info['model_has_names'] = hasattr(model.model, 'names')
                    model_info['model_names_content'] = getattr(model.model, 'names', None) if hasattr(model.model, 'names') else None

                    # 检查model.model的各层
                    if hasattr(model.model, '__iter__'):
                        try:
                            layers_info = []
                            for i, layer in enumerate(model.model):
                                layer_info = {
                                    'index': i,
                                    'type': type(layer).__name__,
                                    'has_names': hasattr(layer, 'names'),
                                    'has_nc': hasattr(layer, 'nc'),
                                    'has_anchors': hasattr(layer, 'anchors')
                                }
                                if hasattr(layer, 'names'):
                                    layer_info['names'] = layer.names
                                if hasattr(layer, 'nc'):
                                    layer_info['nc'] = layer.nc
                                layers_info.append(layer_info)
                            model_info['layers'] = layers_info[-5:]  # 只保留最后5层
                        except:
                            pass

                debug_info['model_analysis'] = model_info

            # 搜索所有可能包含类别信息的位置
            def search_names_recursive(obj, path=""):
                try:
                    if hasattr(obj, 'names') and obj.names:
                        debug_info['potential_class_info'].append({
                            'path': path + '.names',
                            'content': obj.names,
                            'type': type(obj.names).__name__
                        })

                    # 搜索字典类型的对象
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if 'names' in str(k).lower():
                                debug_info['potential_class_info'].append({
                                    'path': f"{path}['{k}']",
                                    'content': v,
                                    'type': type(v).__name__
                                })
                            if hasattr(v, '__dict__') and len(path.split('.')) < 3:  # 限制递归深度
                                search_names_recursive(v, f"{path}['{k}']")

                    # 搜索有__dict__属性的对象
                    elif hasattr(obj, '__dict__') and len(path.split('.')) < 3:
                        for attr_name in dir(obj):
                            if not attr_name.startswith('_'):
                                try:
                                    attr_value = getattr(obj, attr_name)
                                    if 'names' in attr_name.lower() or hasattr(attr_value, 'names'):
                                        search_names_recursive(attr_value, f"{path}.{attr_name}")
                                except:
                                    pass
                except:
                    pass

            # 执行递归搜索
            search_names_recursive(checkpoint, "checkpoint")

            return debug_info

        except Exception as e:
            return {
                'error': f"调试失败: {str(e)}",
                'file_path': weights_file_path
            }

    def debug_mongodb_weight_structure(self, weight_id: str) -> Dict[str, Any]:
        """调试MongoDB中权重文件结构"""
        try:
            # 从MongoDB获取权重记录
            weight_record = self.mongodb.get_weight_by_id(weight_id)
            if not weight_record:
                return {'error': '权重记录不存在'}

            # 将权重数据写入临时文件
            temp_weights_path = os.path.join(self.weights_path, f"debug_{weight_id}.pt")
            os.makedirs(os.path.dirname(temp_weights_path), exist_ok=True)

            with open(temp_weights_path, 'wb') as f:
                f.write(weight_record['weights_data'])

            # 调试权重文件结构
            debug_info = self.debug_weight_file_structure(temp_weights_path)

            # 添加MongoDB记录信息
            debug_info['mongodb_record'] = {
                'weight_id': weight_record['weight_id'],
                'model_name': weight_record['model_name'],
                'model_type': weight_record['model_type'],
                'file_size': weight_record['file_size'],
                'stored_class_count': weight_record.get('model_info', {}).get('class_count', 0),
                'stored_classes': weight_record.get('model_info', {}).get('classes', {}),
                'upload_time': weight_record['upload_time'].isoformat() if weight_record.get('upload_time') else None
            }

            # 清理临时文件
            try:
                os.remove(temp_weights_path)
            except:
                pass

            return debug_info

        except Exception as e:
            return {'error': f'调试MongoDB权重结构失败: {str(e)}'}

    def repair_all_weights_class_info(self) -> Dict[str, Any]:
        """修复所有权重文件的类别信息"""
        try:
            weights_list = self.mongodb.list_weight_files()
            repair_results = {
                'total_weights': len(weights_list),
                'repaired_count': 0,
                'failed_count': 0,
                'details': []
            }

            for weight in weights_list:
                weight_id = weight['weight_id']
                model_name = weight['model_name']

                try:
                    print(f"🔧 检查权重: {model_name} ({weight_id})")

                    # 检查是否需要修复
                    current_class_count = weight.get('model_info', {}).get('class_count', 0)

                    if current_class_count == 0:
                        print(f"🔧 需要修复权重: {model_name}")

                        # 从MongoDB获取完整记录（包含二进制数据）
                        weight_record = self.mongodb.get_weight_by_id(weight_id)
                        if not weight_record:
                            continue

                        # 写入临时文件
                        temp_path = os.path.join(self.weights_path, f"repair_{weight_id}.pt")
                        with open(temp_path, 'wb') as f:
                            f.write(weight_record['weights_data'])

                        # 重新提取模型信息
                        new_model_info = self._extract_model_info(temp_path)

                        # 更新数据库
                        if new_model_info['class_count'] > 0:
                            success = self.mongodb.update_weight_info(weight_id, {
                                'model_info': new_model_info
                            })

                            if success:
                                repair_results['repaired_count'] += 1
                                repair_results['details'].append({
                                    'weight_id': weight_id,
                                    'model_name': model_name,
                                    'status': 'repaired',
                                    'old_class_count': current_class_count,
                                    'new_class_count': new_model_info['class_count']
                                })
                                print(f"✅ 修复成功: {model_name}, 类别数: {new_model_info['class_count']}")
                            else:
                                repair_results['failed_count'] += 1
                                repair_results['details'].append({
                                    'weight_id': weight_id,
                                    'model_name': model_name,
                                    'status': 'update_failed',
                                    'error': '更新数据库失败'
                                })
                        else:
                            repair_results['failed_count'] += 1
                            repair_results['details'].append({
                                'weight_id': weight_id,
                                'model_name': model_name,
                                'status': 'extraction_failed',
                                'error': '无法提取类别信息'
                            })

                        # 清理临时文件
                        try:
                            os.remove(temp_path)
                        except:
                            pass

                    else:
                        repair_results['details'].append({
                            'weight_id': weight_id,
                            'model_name': model_name,
                            'status': 'no_repair_needed',
                            'class_count': current_class_count
                        })

                except Exception as e:
                    repair_results['failed_count'] += 1
                    repair_results['details'].append({
                        'weight_id': weight_id,
                        'model_name': model_name,
                        'status': 'error',
                        'error': str(e)
                    })
                    print(f"❌ 修复失败: {model_name}, 错误: {str(e)}")

            return repair_results

        except Exception as e:
            return {
                'error': f'批量修复失败: {str(e)}',
                'total_weights': 0,
                'repaired_count': 0,
                'failed_count': 0
            }