# models/depth_model.py

import torch
import numpy as np
import cv2
import yaml
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