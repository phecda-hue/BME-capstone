# models/pose_model.py

import numpy as np
from ultralytics import YOLO


# YOLOv11-pose keypoint 인덱스 정의
KP = {
    "nose":           0,
    "left_eye":       1,
    "right_eye":      2,
    "left_ear":       3,
    "right_ear":      4,
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":   10,
    "left_hip":      11,
    "right_hip":     12,
    "left_knee":     13,
    "right_knee":    14,
    "left_ankle":    15,
    "right_ankle":   16,
}


class PoseModel:
    def __init__(self, model_cfg: dict):
        device = model_cfg["device"]
        cfg    = model_cfg["pose"]

        model_source = cfg.get("checkpoint") or cfg["model_id"]

        self.model          = YOLO(model_source)
        self.device         = device
        self.input_size     = cfg["input_size"]
        self.conf_threshold = cfg["conf_threshold"]
        self.iou_threshold  = cfg["iou_threshold"]
        self.fp16           = cfg.get("fp16", False) and device == "cuda"

    def infer(self, frame: np.ndarray) -> list[dict]:
        """
        입력: BGR numpy 배열 (H, W, 3)
        출력: 감지된 사람 리스트
          [
            {
              "bbox":      [x1, y1, x2, y2],   # 픽셀 좌표
              "conf":      float,               # 감지 신뢰도
              "keypoints": np.ndarray,          # (17, 3) — x, y, confidence
            },
            ...
          ]
        """
        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )

        persons = []
        for result in results:
            if result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
            confs = result.boxes.conf.cpu().numpy()       # (N,)
            kpts  = result.keypoints.data.cpu().numpy()   # (N, 17, 3)

            for bbox, conf, kp in zip(boxes, confs, kpts):
                persons.append({
                    "bbox":      bbox.tolist(),
                    "conf":      float(conf),
                    "keypoints": kp,   # (17, 3): [x, y, confidence]
                })

        return persons

    @staticmethod
    def get_keypoint(keypoints: np.ndarray, name: str) -> tuple:
        """
        특정 keypoint의 (x, y, confidence) 반환
        사용 예: x, y, conf = PoseModel.get_keypoint(kp, "nose")
        """
        idx = KP[name]
        return keypoints[idx]

    @staticmethod
    def head_center(keypoints: np.ndarray) -> np.ndarray:
        """
        두부 중심 좌표 반환 (nose + 양쪽 귀 평균)
        confidence가 낮은 keypoint는 제외
        """
        candidates = ["nose", "left_ear", "right_ear"]
        points = []
        for name in candidates:
            x, y, c = PoseModel.get_keypoint(keypoints, name)
            if c > 0.3:   # 신뢰도 0.3 미만은 무시
                points.append([x, y])

        if not points:
            return None

        return np.mean(points, axis=0)   # [x, y]

    @staticmethod
    def shoulder_center(keypoints: np.ndarray) -> np.ndarray:
        """어깨 중심 좌표 반환"""
        l = PoseModel.get_keypoint(keypoints, "left_shoulder")
        r = PoseModel.get_keypoint(keypoints, "right_shoulder")

        valid = [p[:2] for p in [l, r] if p[2] > 0.3]
        if not valid:
            return None

        return np.mean(valid, axis=0)

    @staticmethod
    def hip_center(keypoints: np.ndarray) -> np.ndarray:
        """골반 중심 좌표 반환"""
        l = PoseModel.get_keypoint(keypoints, "left_hip")
        r = PoseModel.get_keypoint(keypoints, "right_hip")

        valid = [p[:2] for p in [l, r] if p[2] > 0.3]
        if not valid:
            return None

        return np.mean(valid, axis=0)