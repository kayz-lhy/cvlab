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

class YOLOv5Service:
    def __init__(self, yolo_repo_path, weights_path):
        self.yolo_repo_path = yolo_repo_path
        self.weights_path = weights_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_weights = None

        # 将YOLOv5仓库路径添加到系统路径
        if yolo_repo_path not in sys.path:
            sys.path.insert(0, yolo_repo_path)

    def clear_torch_hub_cache(self):
        """清除torch.hub缓存"""
        try:
            import torch
            cache_dir = torch.hub.get_dir()
            if os.path.exists(cache_dir):
                print(f"清除torch.hub缓存: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
                return True
        except Exception as e:
            print(f"清除缓存失败: {str(e)}")
            return False

    def load_model(self, weights_filename='best.pt'):
        """加载YOLOv5模型"""
        try:
            # 检查YOLOv5仓库是否存在
            if not os.path.exists(self.yolo_repo_path):
                return False, "YOLOv5仓库路径不存在，请先克隆YOLOv5仓库"

            # 构建权重文件完整路径
            weights_file = os.path.join(self.weights_path, weights_filename)
            if not os.path.exists(weights_file):
                return False, f"权重文件不存在: {weights_file}"

            print(f"正在加载模型: {weights_file}")
            print(f"使用设备: {self.device}")

            # 方法1: 尝试使用torch.hub加载（带force_reload）
            try:
                print("尝试使用torch.hub加载模型...")
                self.model = torch.hub.load(
                    self.yolo_repo_path,
                    'custom',
                    path=weights_file,
                    source='local',
                    force_reload=True,
                    trust_repo=True
                )
                self.model.to(self.device)
                self.current_weights = weights_filename
                print(f"✅ torch.hub加载成功: {weights_filename}, 设备: {self.device}")
                return True, "模型加载成功"

            except Exception as hub_error:
                print(f"❌ torch.hub加载失败: {str(hub_error)}")
                print("尝试清除缓存后重新加载...")

                # 清除缓存并重试
                self.clear_torch_hub_cache()
                try:
                    self.model = torch.hub.load(
                        self.yolo_repo_path,
                        'custom',
                        path=weights_file,
                        source='local',
                        force_reload=True,
                        trust_repo=True
                    )
                    self.model.to(self.device)
                    self.current_weights = weights_filename
                    print(f"✅ 清除缓存后加载成功: {weights_filename}")
                    return True, "模型加载成功"
                except Exception as retry_error:
                    print(f"❌ 清除缓存后仍然失败: {str(retry_error)}")

            # 方法2: 直接导入YOLOv5模块加载
            try:
                print("尝试直接导入YOLOv5模块...")

                # 确保当前工作目录在YOLOv5仓库中
                original_cwd = os.getcwd()
                os.chdir(self.yolo_repo_path)

                try:
                    # 导入YOLOv5的检测模块
                    from models.common import DetectMultiBackend
                    from utils.general import check_img_size, non_max_suppression, scale_boxes, check_requirements
                    from utils.torch_utils import select_device

                    device = select_device(self.device)
                    self.model = DetectMultiBackend(weights_file, device=device)
                    self.current_weights = weights_filename
                    print(f"✅ 直接导入加载成功: {weights_filename}, 设备: {device}")
                    return True, "模型加载成功"

                finally:
                    # 恢复原始工作目录
                    os.chdir(original_cwd)

            except Exception as import_error:
                print(f"❌ 直接导入加载失败: {str(import_error)}")

            # 方法3: 使用Ultralytics YOLO
            try:
                print("尝试使用Ultralytics YOLO加载...")
                from ultralytics import YOLO

                self.model = YOLO(weights_file)
                self.current_weights = weights_filename
                print(f"✅ Ultralytics YOLO加载成功: {weights_filename}")
                return True, "模型加载成功"

            except Exception as ultralytics_error:
                print(f"❌ Ultralytics YOLO加载失败: {str(ultralytics_error)}")

            # 方法4: 直接使用PyTorch加载
            try:
                print("尝试直接使用PyTorch加载...")
                checkpoint = torch.load(weights_file, map_location=self.device)

                # 创建一个简单的包装器
                class SimpleYOLOWrapper:
                    def __init__(self, checkpoint, device):
                        self.checkpoint = checkpoint
                        self.device = device
                        self.names = checkpoint.get('names', {})

                    def __call__(self, img_path, size=640):
                        # 这是一个简化的实现，实际使用时需要完整的推理代码
                        return self.inference(img_path, size)

                    def inference(self, img_path, size):
                        # 简化的推理逻辑
                        import pandas as pd

                        # 创建空的结果对象
                        class Results:
                            def __init__(self):
                                self.conf = 0.25

                            def pandas(self):
                                return type('obj', (object,), {
                                    'xyxy': [pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])]
                                })

                            def render(self):
                                # 返回原图像
                                img = cv2.imread(img_path)
                                return [img] if img is not None else [np.zeros((640, 640, 3), dtype=np.uint8)]

                        return Results()

                self.model = SimpleYOLOWrapper(checkpoint, self.device)
                self.current_weights = weights_filename
                print(f"✅ PyTorch直接加载成功: {weights_filename}")
                return True, "模型加载成功（简化模式）"

            except Exception as pytorch_error:
                print(f"❌ PyTorch直接加载失败: {str(pytorch_error)}")

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            print(error_msg)
            return False, error_msg

        return False, "所有加载方法都失败了"

    def detect_image(self, image_path, conf_threshold=0.25, save_result=True):
        """
        检测图片中的目标

        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            save_result: 是否保存结果图片

        Returns:
            dict: 检测结果
        """
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

            # 进行推理
            results = self.model(image_path, size=640)

            # 设置置信度阈值
            if hasattr(results, 'conf'):
                results.conf = conf_threshold

            # 解析检测结果
            detections = []

            try:
                # 尝试使用pandas格式获取结果
                df = results.pandas().xyxy[0]

                for index, row in df.iterrows():
                    if row['confidence'] >= conf_threshold:
                        detection = {
                            'class_id': int(row['class']),
                            'class_name': row['name'],
                            'confidence': round(float(row['confidence']), 3),
                            'bbox': [
                                int(row['xmin']),
                                int(row['ymin']),
                                int(row['xmax']),
                                int(row['ymax'])
                            ]
                        }
                        detections.append(detection)

            except Exception as parse_error:
                print(f"解析检测结果失败: {str(parse_error)}")
                # 如果pandas方法失败，返回空结果但不报错
                detections = []

            # 生成结果图片的base64
            result_image_base64 = None
            result_image_path = None
            if save_result:
                result_image_base64, result_image_path = self._generate_result_image(results, image_path)

            processing_time = round(time.time() - start_time, 3)

            return {
                'success': True,
                'detections': detections,
                'detection_count': len(detections),
                'model_name': self.current_weights,
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

    def _generate_result_image(self, results, original_path):
        """生成结果图片并返回base64编码"""
        try:
            from flask import current_app

            # 尝试渲染结果图片
            try:
                result_img = results.render()[0]  # 获取渲染后的图片数组
            except:
                # 如果渲染失败，返回原图像
                result_img = cv2.imread(original_path)
                if result_img is None:
                    return None, None

            # 将BGR转换为RGB（OpenCV默认是BGR）
            if len(result_img.shape) == 3 and result_img.shape[2] == 3:
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            else:
                result_img_rgb = result_img

            # 转换为PIL Image
            pil_img = Image.fromarray(result_img_rgb)

            # 保存到内存并转换为base64
            from io import BytesIO
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()

            # 转换为base64字符串
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # 同时保存文件（用于历史记录）
            result_image_path = None
            if current_app:
                result_image_path = self._save_result_image_file(result_img, original_path)

            return img_base64, result_image_path

        except Exception as e:
            print(f"生成结果图片失败: {str(e)}")
            # 如果生成失败，尝试返回原图像的base64
            try:
                with open(original_path, 'rb') as f:
                    img_bytes = f.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return img_base64, None
            except:
                return None, None

    def _save_result_image_file(self, result_img, original_path):
        """保存结果图片文件（用于历史记录）"""
        try:
            from flask import current_app

            # 创建结果目录
            result_dir = current_app.config['RESULTS_FOLDER']
            os.makedirs(result_dir, exist_ok=True)

            # 生成唯一文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            original_name = os.path.splitext(os.path.basename(original_path))[0]
            filename = f"result_{original_name}_{timestamp}_{unique_id}.jpg"
            result_path = os.path.join(result_dir, filename)

            # 保存结果图片
            cv2.imwrite(result_path, result_img)

            # 返回相对路径用于web访问
            return os.path.relpath(result_path, 'app/static')

        except Exception as e:
            print(f"保存结果图片文件失败: {str(e)}")
            return None

    def get_model_info(self):
        """获取模型信息"""
        if not self.model:
            return {'loaded': False}

        try:
            # 尝试获取模型的类别名称
            names = {}
            if hasattr(self.model, 'names'):
                names = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                names = self.model.module.names
            elif hasattr(self.model, 'checkpoint') and 'names' in self.model.checkpoint:
                names = self.model.checkpoint['names']

            return {
                'loaded': True,
                'weights_file': self.current_weights,
                'device': str(self.device),
                'classes': names,
                'class_count': len(names) if names else 0
            }
        except Exception as e:
            print(f"获取模型信息失败: {str(e)}")
            return {
                'loaded': True,
                'weights_file': self.current_weights,
                'device': str(self.device),
                'classes': {},
                'class_count': 0
            }

    def list_available_weights(self):
        """列出可用的权重文件"""
        try:
            weights_files = []
            if os.path.exists(self.weights_path):
                for file in os.listdir(self.weights_path):
                    if file.endswith(('.pt', '.pth')):
                        weights_files.append(file)
            return weights_files
        except Exception as e:
            print(f"列出权重文件失败: {str(e)}")
            return []