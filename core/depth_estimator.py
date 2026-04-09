# core/depth_estimator.py

"""
깊이맵 후처리 모듈
==================
DepthModel.infer() 의 원시 깊이맵에는 두 종류의 노이즈가 있음

  1. 공간 노이즈  : 한 프레임 안에서 인접 픽셀 간 깊이값 튐
                    → DepthModel.get_depth_at() 의 패치 중앙값으로 부분 처리됨
                    → 이 모듈에서는 다루지 않음

  2. 시간 노이즈  : 같은 픽셀의 깊이값이 프레임마다 ±10~20% 흔들림
                    Depth Anything v2 는 단안 카메라 기반으로,
                    프레임 간 일관성이 완벽하지 않음
                    → 이 모듈에서 지수 이동 평균(EMA) 으로 안정화

  3. 이상값 제거  : 이전 프레임 대비 깊이가 급격히 변한 픽셀을 이전 값으로 대체
                    슬라이드 구조물 반사, 물보라 등으로 인한 스파이크 제거

사용 예:
    estimator = DepthEstimator(cfg)

    # 매 프레임
    raw_depth    = depth_model.infer(frame)
    smooth_depth = estimator.update(raw_depth)
"""

import numpy as np


class DepthEstimator:
    """
    깊이맵 시간축 평활화 및 이상값 제거

    알고리즘:
      1. 이상값 마스킹  : |depth - prev_depth| > spike_threshold 인 픽셀을 이전값으로 대체
      2. EMA 평활화    : smooth = α × raw + (1 - α) × prev_smooth
         α(ema_alpha) 가 클수록 현재 프레임 반영 비중 증가 (응답성 ↑, 안정성 ↓)
         α 가 작을수록 이전 프레임 반영 비중 증가 (안정성 ↑, 응답성 ↓)
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: model.yaml 의 depth_estimator 섹션 dict
                 아래 키를 사용
                   ema_alpha       : EMA 가중치 (0.0 ~ 1.0, 권장 0.3 ~ 0.5)
                   spike_threshold : 이상값 판정 깊이 변화량 임계값 (m)
        """
        self.ema_alpha:       float = cfg.get("ema_alpha",       0.4)
        self.spike_threshold: float = cfg.get("spike_threshold", 1.0)

        # 이전 프레임 평활화 결과 (첫 프레임 전까지 None)
        self._prev_smooth: np.ndarray | None = None

    # ─────────────────────────────────────────
    # 퍼블릭 메서드
    # ─────────────────────────────────────────

    def update(self, raw_depth: np.ndarray) -> np.ndarray:
        """
        원시 깊이맵을 받아 후처리된 깊이맵을 반환

        Args:
            raw_depth: DepthModel.infer() 반환값 (H, W), float32, 단위 m

        Returns:
            후처리된 깊이맵 (H, W), float32, 단위 m
            첫 프레임은 raw_depth 를 그대로 반환
        """
        # 첫 프레임 — 비교 대상 없으므로 그대로 저장 후 반환
        if self._prev_smooth is None:
            self._prev_smooth = raw_depth.copy()
            return raw_depth.copy()

        # ── 1단계: 이상값 마스킹 ─────────────────────────────────────────
        # 이전 평활화 값 대비 급격히 변한 픽셀 → 이전값으로 대체
        # 물보라, 슬라이드 구조물 반사 등으로 인한 순간 스파이크 제거
        spike_mask    = np.abs(raw_depth - self._prev_smooth) > self.spike_threshold
        cleaned_depth = np.where(spike_mask, self._prev_smooth, raw_depth)

        # ── 2단계: EMA 평활화 ─────────────────────────────────────────────
        # smooth = α × cleaned + (1 - α) × prev_smooth
        smooth_depth = (
            self.ema_alpha * cleaned_depth
            + (1.0 - self.ema_alpha) * self._prev_smooth
        )

        self._prev_smooth = smooth_depth
        return smooth_depth.astype(np.float32)

    def reset(self) -> None:
        """
        상태 초기화
        카메라 전환, 장면이 크게 바뀌는 경우 호출
        """
        self._prev_smooth = None

    # ─────────────────────────────────────────
    # 파라미터 정보 조회
    # ─────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """첫 프레임 처리 여부"""
        return self._prev_smooth is not None

    def get_params(self) -> dict:
        """현재 파라미터 반환 (디버그용)"""
        return {
            "ema_alpha":       self.ema_alpha,
            "spike_threshold": self.spike_threshold,
        }