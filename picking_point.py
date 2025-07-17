# picking_point.py

import os
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from ultralytics import YOLO
from sklearn.decomposition import PCA
from scipy.stats import skew

class PickingProcessor:
    """
    YOLO+SAM을 이용한 2D 객체 분할과
    입력된 3D 포인트 클라우드의 PCA 분석을 모두 수행하는 클래스.
    """
    def __init__(self, sam_checkpoint_name: str = "sam_vit_h.pth", yolo_model_name: str = "yolov8m.pt"):
        self.sam_checkpoint_name = sam_checkpoint_name
        self.yolo_model_name = yolo_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_sam = None
        self.model_yolo = None
        self._load_models()

    def _load_models(self):
        """YOLO와 SAM 모델을 로드합니다."""
        try:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = os.getcwd()

            sam_model_path = os.path.join(script_dir, self.sam_checkpoint_name)
            if not os.path.exists(sam_model_path):
                raise FileNotFoundError(f"오류: SAM 모델 '{sam_model_path}'을 찾을 수 없습니다.")

            sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
            sam.to(self.device)
            self.predictor_sam = SamPredictor(sam)
            self.model_yolo = YOLO(self.yolo_model_name)
            print("AI 모델 로드 완료.")
        except Exception as e:
            print(f"AI 모델 로드 중 오류 발생: {e}")
            exit(1)

    def detect_and_segment(self, rgb_image: np.ndarray):
        """주어진 RGB 이미지에서 객체를 탐지하고 분할 마스크를 반환합니다."""
        self.predictor_sam.set_image(rgb_image)
        results = self.model_yolo(rgb_image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        detections = []
        if boxes.size == 0:
            return detections

        for box in boxes:
            masks, scores, _ = self.predictor_sam.predict(box=np.array(box), multimask_output=True)
            mask = masks[np.argmax(scores)]
            detections.append({'box': box, 'mask': mask})
            
        return detections

    def analyze_point_cloud(self, points: np.ndarray):
        """
        주어진 3D 포인트 클라우드를 PCA로 분석하여 중심과 방향을 찾습니다.

        Args:
            points (np.ndarray): (N, 3) 형태의 3D 포인트 배열.

        Returns:
            dict: 중심, 방향, 왜도 등 분석 결과가 담긴 딕셔너리.
        """
        if points is None or len(points) < 10:
            return None
        
        try:
            pca = PCA(n_components=3)
            pca.fit(points)
            
            center = pca.mean_
            direction = pca.components_[0] # 제1 주성분
            
            # 피킹 방향 결정을 위한 왜도 계산
            projections = np.dot(points - center, direction)
            s = skew(projections)
            
            # 왜도(skewness)가 크면(>0.1) 분포가 한쪽으로 치우쳐 있다는 의미.
            # 왜도가 양수이면 데이터가 오른쪽으로, 음수이면 왼쪽으로 긴 꼬리를 가짐.
            # 이를 이용해 파지 방향을 결정 (-np.sign(s)).
            arrow_direction = -np.sign(s) * direction if np.abs(s) >= 0.1 else direction
            
            return {
                'center_cam': center,
                'direction': direction,
                'arrow_direction': arrow_direction,
                'projections': projections,
                'skew_value': s
            }
        except Exception as e:
            print(f"PCA 분석 중 오류 발생: {e}")
            return None