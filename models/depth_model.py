# models/depth_model.py

import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthModel:
    def __init__(self, model_cfg: dict):
        self.device = torch.device(model_cfg["device"])
        cfg = model_cfg["depth"]

        self.input_size = cfg["input_size"]
        self.max_depth  = cfg["max_depth"]
        self.fp16       = cfg.get("fp16", False) and self.device.type == "cuda"

        # 모델 로드 (로컬 checkpoint 우선, 없으면 HuggingFace 자동 다운로드)
        model_source = cfg.get("checkpoint") or cfg["model_id"]

        self.processor = AutoImageProcessor.from_pretrained(model_source)
        self.model     = AutoModelForDepthEstimation.from_pretrained(model_source)

        self.model.to(self.device)
        self.model.eval()

        if self.fp16:
            self.model.half()

    # ─────────────────────────────────────────
    # 깊이맵 추정
    # ─────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        입력: BGR numpy 배열 (H, W, 3)
        출력: 미터 단위 깊이맵 numpy 배열 (H, W), float32
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(
            images=rgb,
            return_tensors="pt",
            size={"height": self.input_size, "width": self.input_size},
        )

        if self.fp16:
            inputs = {k: v.half() for k, v in inputs.items()}

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 원본 해상도로 복원
        depth = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(frame.shape[0], frame.shape[1])],
        )[0]["predicted_depth"]

        depth_np = depth.cpu().float().numpy()

        # max_depth 클리핑 및 음수 제거
        depth_np = np.clip(depth_np, 0.0, self.max_depth)

        return depth_np.astype(np.float32)

    # ─────────────────────────────────────────
    # 깊이 조회
    # ─────────────────────────────────────────

    @staticmethod
    def get_depth_at(
        depth_map: np.ndarray,
        x: float,
        y: float,
        patch_radius: int = 2,
    ) -> float | None:
        """
        픽셀 좌표 (x, y) 주변 패치의 중앙값으로 깊이 반환 (단위: 미터)
        단순 픽셀 조회 대신 패치 중앙값을 사용하여 Depth Anything의
        프레임 간 노이즈에 강건하게 대응

        Args:
            depth_map:    infer() 반환값 (H, W), float32
            x, y:         픽셀 좌표 (float 허용, 내부에서 int 변환)
            patch_radius: 중앙값을 구할 패치 반경 (기본 2 → 5×5 패치)

        Returns:
            깊이값 (m), 좌표가 범위 밖이면 None
        """
        h, w = depth_map.shape
        xi, yi = int(round(x)), int(round(y))

        # 범위 검사
        if not (0 <= xi < w and 0 <= yi < h):
            return None

        # 패치 범위 클리핑 (이미지 경계 처리)
        y0 = max(0, yi - patch_radius)
        y1 = min(h, yi + patch_radius + 1)
        x0 = max(0, xi - patch_radius)
        x1 = min(w, xi + patch_radius + 1)

        patch = depth_map[y0:y1, x0:x1]

        # 유효값(0보다 큰 픽셀)만 중앙값 계산
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    @staticmethod
    def get_depth_at_keypoint(
        depth_map: np.ndarray,
        keypoint: np.ndarray,
        patch_radius: int = 2,
    ) -> float | None:
        """
        keypoint 배열 [x, y] 또는 [x, y, conf]에서 바로 깊이 조회
        pose_model.py의 반환값과 직접 연결되도록 설계

        Args:
            depth_map: infer() 반환값 (H, W), float32
            keypoint:  [x, y] 또는 [x, y, conf] numpy 배열
                       (head_center, hip_center 등의 반환값 그대로 사용 가능)

        Returns:
            깊이값 (m), 조회 실패 시 None
        """
        if keypoint is None:
            return None

        return DepthModel.get_depth_at(
            depth_map,
            x=float(keypoint[0]),
            y=float(keypoint[1]),
            patch_radius=patch_radius,
        )

    # ─────────────────────────────────────────
    # 실제 속도 추정
    # ─────────────────────────────────────────

    @staticmethod
    def get_real_velocity(
        dx_px: float,
        dy_px: float,
        depth_m: float,
        fx: float,
        fy: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """
        픽셀 변위 → 실제 속도 변환 (핀홀 카메라 모델 기반)

        핀홀 카메라 역투영 공식:
            X_real = (x_px - cx) * depth / fx  →  ΔX = Δx_px * depth / fx
            Y_real = (y_px - cy) * depth / fy  →  ΔY = Δy_px * depth / fy

        cx, cy는 변위(Δ) 계산 시 상쇄되므로 불필요

        Args:
            dx_px:   x 방향 픽셀 변위 (현재 - 이전)
            dy_px:   y 방향 픽셀 변위 (현재 - 이전)
            depth_m: 해당 keypoint의 깊이 (미터)
            fx, fy:  카메라 초점거리 (픽셀 단위, camera.yaml의 intrinsics)
            dt:      프레임 간격 (초), 예: 0.1 (10 FPS)

        Returns:
            (vx, vy, speed) 단위 모두 m/s
            vx:    수평 실제 속도
            vy:    수직 실제 속도 (양수 = 화면 아래 방향)
            speed: 2D 속력 sqrt(vx^2 + vy^2)
        """
        if depth_m <= 0 or dt <= 0:
            return 0.0, 0.0, 0.0

        vx = (dx_px * depth_m / fx) / dt   # m/s
        vy = (dy_px * depth_m / fy) / dt   # m/s
        speed = float(np.sqrt(vx**2 + vy**2))

        return float(vx), float(vy), speed