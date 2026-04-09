# core/impact_estimator.py

"""
머리 충격량 추정 및 위험도 판정
================================
FallDetector 가 낙상을 확정한 후,
FallResult.head_velocity 를 받아 위험도를 계산

물리 공식: I = m × Δv
  Δv = |prev_head_velocity - head_velocity|
  → 바닥 충돌 순간 속도가 급감 → Δv 증가 → 충격량 증가

사용 예:
    fall_results   = fall_detector.update(persons, depth_map, dt)
    impact_results = impact_estimator.update(fall_results, persons, depth_map)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from models.pose_model import PoseModel
from models.depth_model import DepthModel


# ─────────────────────────────────────────
# 열거형 정의
# ─────────────────────────────────────────

class DangerLevel(Enum):
    """
    낙상 후 위험도 3단계
    머리 충격량(impulse = m × Δv)으로 판정
    """
    LOW     = "저위험"
    CAUTION = "주의"
    DANGER  = "위험"


# ─────────────────────────────────────────
# 트랙 ID별 충격 관련 상태
# ─────────────────────────────────────────

@dataclass
class ImpactState:
    """
    트랙 ID별로 이전 프레임의 head 속도를 저장
    FallDetector 의 PersonState 와 별도로 관리
    """
    track_id: int

    # 이전 프레임 head 실제 속도 (m/s)
    # Δv = |prev_head_velocity - head_velocity| 계산에 사용
    prev_head_velocity: float = 0.0


# ─────────────────────────────────────────
# 충격 추정 결과
# ─────────────────────────────────────────

@dataclass
class ImpactResult:
    """
    ImpactEstimator.update() 반환값
    낙상이 감지되지 않은 경우 danger_level=None, impulse=0.0
    """
    track_id:     int
    danger_level: DangerLevel | None   # 낙상 미감지 시 None
    impulse:      float = 0.0          # 머리 충격량 (kg·m/s)
    delta_v:      float = 0.0          # 속도 변화량 (m/s), 디버그용


# ─────────────────────────────────────────
# 충격 추정기 메인 클래스
# ─────────────────────────────────────────

class ImpactEstimator:
    """
    FallDetector 로부터 낙상 결과를 받아 머리 충격량과 위험도를 판정

    분리 이유:
      - FallDetector  : 낙상 여부 판정 (운동학적 조건)
      - ImpactEstimator: 위험도 판정 (물리적 충격량)
      두 책임을 분리하여 각각 독립적으로 임계값 조정 가능

    사용 예:
        estimator      = ImpactEstimator(threshold_cfg)
        impact_results = estimator.update(fall_results, persons, depth_map)
    """

    # 성인 기준 L_factor 픽셀값 (fall_detector.py 의 _REFERENCE_L 과 동일)
    _REFERENCE_L = 150.0

    def __init__(self, threshold_cfg: dict):
        """
        Args:
            threshold_cfg: thresholds.yaml 전체 dict
        """
        dl = threshold_cfg["danger_level"]

        # 성인 두부 기준 질량 (kg)
        # 실제 사용 질량은 L_factor 로 보정됨
        # estimated_mass = head_mass_kg × (l_factor / REFERENCE_L)
        self.head_mass_kg: float = dl["head_mass_kg"]

        # 머리 위치 임계값 (픽셀)
        # head_y > floor_y - floor_margin 일 때만 충격량 계산
        self.floor_y:      float = dl["floor_y"]
        self.floor_margin: float = dl["floor_margin"]

        # 충격량(impulse = estimated_mass × Δv, kg·m/s) 임계값
        # L_factor 보정된 질량을 사용하므로 delta_v 가 아닌 impulse 기준으로 판정
        self.impulse_caution: float = dl["impulse_caution"]
        self.impulse_danger:  float = dl["impulse_danger"]

        # ── 트랙 ID별 상태 저장 ──────────────────
        # {track_id: ImpactState}
        self._states: dict[int, ImpactState] = {}

    # ─────────────────────────────────────────
    # 퍼블릭 메서드
    # ─────────────────────────────────────────

    def update(
        self,
        fall_results: list,           # list[FallResult]
        persons:      list[dict],     # PoseModel.infer() 반환값 (keypoints 참조용)
        depth_map:    np.ndarray,     # DepthModel.infer() 반환값 (머리 위치 확인용)
    ) -> list[ImpactResult]:
        """
        낙상 결과를 받아 위험도를 판정

        Args:
            fall_results: FallDetector.update() 반환값 (list[FallResult])
            persons:      PoseModel.infer() 반환값 — 머리 픽셀 좌표 조회용
            depth_map:    DepthModel.infer() 반환값 — 머리 위치(y) 확인용

        Returns:
            fall_results 와 같은 순서의 ImpactResult 리스트
        """
        results = []
        active_ids = set()

        for fall_result, person in zip(fall_results, persons):
            tid   = fall_result.track_id
            state = self._get_or_create_state(tid)
            active_ids.add(tid)

            kp   = person["keypoints"]
            head = PoseModel.head_center(kp)

            if fall_result.fall_detected:
                impact_result = self._calc_impact(
                    state, head, fall_result.head_velocity,
                    fall_result.l_factor,
                )
            else:
                # 낙상 미감지 — 위험도 없음, head 속도만 갱신
                impact_result = ImpactResult(
                    track_id     = tid,
                    danger_level = None,
                    impulse      = 0.0,
                    delta_v      = 0.0,
                )

            # head 속도 이력 갱신 (낙상 여부 무관하게 매 프레임 갱신)
            state.prev_head_velocity = fall_result.head_velocity

            results.append(impact_result)

        # 이번 프레임에 없는 트랙 ID 정리
        lost_ids = [tid for tid in self._states if tid not in active_ids]
        for tid in lost_ids:
            del self._states[tid]

        return results

    # ─────────────────────────────────────────
    # 내부 처리 메서드
    # ─────────────────────────────────────────

    def _get_or_create_state(self, track_id: int) -> ImpactState:
        if track_id not in self._states:
            self._states[track_id] = ImpactState(track_id=track_id)
        return self._states[track_id]

    def _calc_impact(
        self,
        state:         ImpactState,
        head:          np.ndarray | None,
        head_velocity: float,
        l_factor:      float | None,
    ) -> "ImpactResult":
        """
        낙상 확정 후 머리 충격량으로 위험도 판정

        두부 질량 추정 [논문 2 L_factor 응용]:
          estimated_mass = head_mass_kg × (l_factor / REFERENCE_L)
          → 어린이(l_factor 작음) → 질량 작게 추정
          → 성인(l_factor 큼)    → 질량 기준값 유지
          → l_factor 없으면 head_mass_kg 고정값 사용

        물리 공식: I = m × Δv
          Δv = |prev_head_velocity - head_velocity|
          바닥 충돌 순간 속도가 0에 가까워지면 Δv 가 커짐

        위험도는 delta_v 가 아닌 impulse(kg·m/s) 기준으로 판정
          → 체형이 반영된 실제 충격량으로 판정

        머리가 floor_y 이상 내려왔을 때만 계산
        그 외(머리를 바닥에 부딪히지 않은 낙상)는 LOW 반환

        Returns:
            ImpactResult
        """
        tid = state.track_id

        # 머리가 바닥 근처에 있는지 확인
        if head is None or float(head[1]) <= self.floor_y - self.floor_margin:
            return ImpactResult(
                track_id     = tid,
                danger_level = DangerLevel.LOW,
                impulse      = 0.0,
                delta_v      = 0.0,
            )

        # ── L_factor 기반 두부 질량 추정 [논문 2 응용] ───────────────────
        if l_factor and l_factor > 0:
            estimated_mass = self.head_mass_kg * (l_factor / self._REFERENCE_L)
        else:
            estimated_mass = self.head_mass_kg   # l_factor 없으면 고정값

        # ── 충격량 계산: I = m × Δv ───────────────────────────────────────
        delta_v = abs(state.prev_head_velocity - head_velocity)
        impulse = estimated_mass * delta_v

        # ── impulse 기준 위험도 판정 ──────────────────────────────────────
        if impulse >= self.impulse_danger:
            level = DangerLevel.DANGER
        elif impulse >= self.impulse_caution:
            level = DangerLevel.CAUTION
        else:
            level = DangerLevel.LOW

        return ImpactResult(
            track_id     = tid,
            danger_level = level,
            impulse      = impulse,
            delta_v      = delta_v,
        )