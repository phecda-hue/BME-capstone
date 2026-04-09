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

# 무게중심(CoG) 계산에 사용할 keypoint 인덱스 (논문 1 기준)
# {nose, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee}
COG_INDICES = [0, 5, 6, 11, 12, 13, 14]


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

    # ─────────────────────────────────────────
    # 단일 keypoint 조회
    # ─────────────────────────────────────────

    @staticmethod
    def get_keypoint(keypoints: np.ndarray, name: str) -> np.ndarray:
        """
        특정 keypoint의 (x, y, confidence) 반환
        사용 예: x, y, conf = PoseModel.get_keypoint(kp, "nose")
        """
        idx = KP[name]
        return keypoints[idx]

    # ─────────────────────────────────────────
    # 신체 부위 중심 좌표 (기존)
    # ─────────────────────────────────────────

    @staticmethod
    def head_center(keypoints: np.ndarray) -> np.ndarray | None:
        """
        두부 중심 좌표 반환 (nose + 양쪽 귀 평균)
        confidence가 낮은 keypoint는 제외
        """
        candidates = ["nose", "left_ear", "right_ear"]
        points = []
        for name in candidates:
            x, y, c = PoseModel.get_keypoint(keypoints, name)
            if c > 0.3:
                points.append([x, y])

        if not points:
            return None

        return np.mean(points, axis=0)   # [x, y]

    @staticmethod
    def shoulder_center(keypoints: np.ndarray) -> np.ndarray | None:
        """어깨 중심 좌표 반환"""
        l = PoseModel.get_keypoint(keypoints, "left_shoulder")
        r = PoseModel.get_keypoint(keypoints, "right_shoulder")

        valid = [p[:2] for p in [l, r] if p[2] > 0.3]
        if not valid:
            return None

        return np.mean(valid, axis=0)

    @staticmethod
    def hip_center(keypoints: np.ndarray) -> np.ndarray | None:
        """골반 중심 좌표 반환"""
        l = PoseModel.get_keypoint(keypoints, "left_hip")
        r = PoseModel.get_keypoint(keypoints, "right_hip")

        valid = [p[:2] for p in [l, r] if p[2] > 0.3]
        if not valid:
            return None

        return np.mean(valid, axis=0)

    # ─────────────────────────────────────────
    # 신체 부위 중심 좌표 (추가)
    # ─────────────────────────────────────────

    @staticmethod
    def feet_center(keypoints: np.ndarray) -> np.ndarray | None:
        """
        발목 중심 좌표 반환 (left_ankle + right_ankle 평균)
        논문 1의 |y_hips - y_feet| < 30 조건에 사용
        """
        l = PoseModel.get_keypoint(keypoints, "left_ankle")
        r = PoseModel.get_keypoint(keypoints, "right_ankle")

        valid = [p[:2] for p in [l, r] if p[2] > 0.3]
        if not valid:
            return None

        return np.mean(valid, axis=0)   # [x, y]

    @staticmethod
    def knee_center(keypoints: np.ndarray) -> np.ndarray | None:
        """
        무릎 중심 좌표 반환 (left_knee + right_knee 평균)
        CoG 계산 및 몸통-다리 각도 계산에 사용
        """
        l = PoseModel.get_keypoint(keypoints, "left_knee")
        r = PoseModel.get_keypoint(keypoints, "right_knee")

        valid = [p[:2] for p in [l, r] if p[2] > 0.3]
        if not valid:
            return None

        return np.mean(valid, axis=0)   # [x, y]

    @staticmethod
    def cog(keypoints: np.ndarray) -> np.ndarray | None:
        """
        무게중심(Center of Gravity) 반환
        논문 1 기준: {nose(0), left_shoulder(5), right_shoulder(6),
                      left_hip(11), right_hip(12), left_knee(13), right_knee(14)}
        confidence 0.3 미만 keypoint는 제외하고 평균 계산
        반환: [x, y] 또는 유효 keypoint가 없으면 None
        """
        points = []
        for idx in COG_INDICES:
            x, y, c = keypoints[idx]
            if c > 0.3:
                points.append([x, y])

        if not points:
            return None

        return np.mean(points, axis=0)   # [x, y]

    @staticmethod
    def l_factor(keypoints: np.ndarray) -> float | None:
        """
        체형 보정 계수 반환 (논문 2 기준)
        왼쪽 어깨(5)와 왼쪽 엉덩이(11) 사이의 유클리드 거리
        어린이/성인 혼재 환경에서 픽셀 기반 임계값을 체형에 맞게 보정할 때 사용

        L_factor = sqrt((x_shoulder - x_hip)^2 + (y_shoulder - y_hip)^2)

        사용 예:
          lf = PoseModel.l_factor(kp)
          if lf and hip_y < feet_y + alpha * lf:
              ...  # 넘어진 상태 판단
        """
        ls = PoseModel.get_keypoint(keypoints, "left_shoulder")
        lh = PoseModel.get_keypoint(keypoints, "left_hip")

        # 두 keypoint 모두 신뢰도 0.3 이상이어야 유효
        if ls[2] < 0.3 or lh[2] < 0.3:
            # 왼쪽이 가려진 경우 오른쪽으로 폴백
            rs = PoseModel.get_keypoint(keypoints, "right_shoulder")
            rh = PoseModel.get_keypoint(keypoints, "right_hip")
            if rs[2] < 0.3 or rh[2] < 0.3:
                return None
            ls, lh = rs, rh

        return float(np.linalg.norm(ls[:2] - lh[:2]))