# core/pose_estimator.py

"""
키포인트 후처리 모듈
====================
PoseModel.infer() 의 원시 keypoints 에는 두 종류의 불안정성이 있음

  1. 좌표 떨림  : 같은 관절의 픽셀 좌표가 프레임마다 수 픽셀씩 흔들림
                  YOLOv11-pose 는 프레임 간 일관성을 보장하지 않음
                  → EMA(지수 이동 평균) 로 안정화

  2. 신뢰도 저하: 가림(슬라이드 구조물), 물보라, 화면 경계 등으로
                  특정 keypoint 의 confidence 가 일시적으로 낮아짐
                  → 신뢰도 임계값 미만이면 이전 프레임 값으로 대체

사용 예:
    estimator = PoseEstimator(cfg)

    # 매 프레임
    raw_persons    = pose_model.infer(frame)
    smooth_persons = estimator.update(raw_persons, track_ids)
"""

import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────
# 트랙 ID별 keypoint 상태
# ─────────────────────────────────────────

@dataclass
class PoseState:
    """
    트랙 ID별로 이전 프레임의 평활화된 keypoints 를 저장
    keypoints shape: (17, 3) — [x, y, confidence]
    """
    track_id: int

    # 이전 프레임 평활화 결과
    # 첫 프레임 전까지 None
    prev_keypoints: np.ndarray | None = None   # (17, 3)


# ─────────────────────────────────────────
# 키포인트 후처리기 메인 클래스
# ─────────────────────────────────────────

class PoseEstimator:
    """
    YOLOv11-pose keypoint 후처리기
    EMA 평활화 + 신뢰도 기반 보간을 적용하여
    안정적인 keypoints 를 반환

    처리 순서 (매 프레임, 사람별):
      1. 신뢰도 임계값 미만 keypoint → 이전 프레임 값으로 대체
      2. EMA 평활화 적용
      3. 평활화된 keypoints 를 persons dict 에 덮어쓰기

    사용 예:
        estimator      = PoseEstimator(cfg)
        smooth_persons = estimator.update(raw_persons, track_ids)
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: model.yaml 의 pose_estimator 섹션 dict
                 아래 키를 사용
                   ema_alpha      : EMA 가중치 (0.0 ~ 1.0, 권장 0.4 ~ 0.6)
                   min_confidence : 보간 적용 신뢰도 임계값 (권장 0.3)
        """
        # EMA 가중치
        # 클수록 현재 프레임 반영 비중 증가 (응답성 ↑, 안정성 ↓)
        # 작을수록 이전 프레임 반영 비중 증가 (안정성 ↑, 응답성 ↓)
        self.ema_alpha: float = cfg.get("ema_alpha", 0.5)

        # 신뢰도 임계값
        # 이 값 미만인 keypoint 는 이전 프레임 값으로 대체
        # PoseModel 내부의 0.3 기준과 동일하게 맞추는 것을 권장
        self.min_confidence: float = cfg.get("min_confidence", 0.3)

        # ── 트랙 ID별 상태 저장 ──────────────────
        # {track_id: PoseState}
        self._states: dict[int, PoseState] = {}

    # ─────────────────────────────────────────
    # 퍼블릭 메서드
    # ─────────────────────────────────────────

    def update(
        self,
        persons:   list[dict],
        track_ids: list[int] | None = None,
    ) -> list[dict]:
        """
        원시 keypoints 를 받아 후처리된 persons 리스트를 반환

        Args:
            persons:   PoseModel.infer() 반환값
                       [{"bbox": [...], "conf": float, "keypoints": ndarray(17,3)}, ...]
            track_ids: persons 와 같은 순서의 트랙 ID 리스트
                       None 이면 리스트 인덱스를 ID 로 사용

        Returns:
            keypoints 가 평활화된 persons 리스트 (원본 dict 는 수정하지 않음)
        """
        if track_ids is None:
            track_ids = list(range(len(persons)))

        smooth_persons = []
        active_ids     = set()

        for person, tid in zip(persons, track_ids):
            active_ids.add(tid)
            state          = self._get_or_create_state(tid)
            smooth_kp      = self._smooth_keypoints(person["keypoints"], state)
            smooth_persons.append({
                **person,
                "keypoints": smooth_kp,   # 평활화된 keypoints 로 교체
            })

        # 이번 프레임에 없는 트랙 ID 상태 정리
        lost_ids = [tid for tid in self._states if tid not in active_ids]
        for tid in lost_ids:
            del self._states[tid]

        return smooth_persons

    def reset(self, track_id: int | None = None) -> None:
        """
        상태 초기화
        track_id 지정 시 해당 ID 만 초기화, None 이면 전체 초기화

        Args:
            track_id: 초기화할 트랙 ID, None 이면 전체
        """
        if track_id is None:
            self._states.clear()
        elif track_id in self._states:
            del self._states[track_id]

    # ─────────────────────────────────────────
    # 내부 처리 메서드
    # ─────────────────────────────────────────

    def _get_or_create_state(self, track_id: int) -> PoseState:
        if track_id not in self._states:
            self._states[track_id] = PoseState(track_id=track_id)
        return self._states[track_id]

    def _smooth_keypoints(
        self,
        raw_kp: np.ndarray,
        state:  PoseState,
    ) -> np.ndarray:
        """
        단일 사람의 keypoints (17, 3) 에 후처리 적용

        처리 순서:
          1. 신뢰도 임계값 미만 keypoint → 이전 프레임 값으로 대체
          2. EMA 평활화 적용 (x, y 좌표만, confidence 는 원본 유지)
          3. 상태 갱신

        Args:
            raw_kp: 원시 keypoints (17, 3) — [x, y, confidence]
            state:  해당 트랙의 PoseState

        Returns:
            평활화된 keypoints (17, 3)
        """
        # 첫 프레임 — 비교 대상 없으므로 그대로 저장 후 반환
        if state.prev_keypoints is None:
            state.prev_keypoints = raw_kp.copy()
            return raw_kp.copy()

        smooth_kp = raw_kp.copy()

        # ── 1단계: 신뢰도 기반 보간 ───────────────────────────────────────
        # 신뢰도 임계값 미만인 keypoint 의 x, y 를 이전 프레임 값으로 대체
        # confidence 값 자체는 현재 프레임 원본을 유지
        # (신뢰도가 낮다는 정보는 하위 모듈에서 활용할 수 있도록 보존)
        low_conf_mask = raw_kp[:, 2] < self.min_confidence   # (17,) bool
        smooth_kp[low_conf_mask, 0] = state.prev_keypoints[low_conf_mask, 0]  # x
        smooth_kp[low_conf_mask, 1] = state.prev_keypoints[low_conf_mask, 1]  # y

        # ── 2단계: EMA 평활화 ─────────────────────────────────────────────
        # x, y 좌표에만 적용, confidence 는 현재 프레임 값 유지
        # smooth_xy = α × current_xy + (1 - α) × prev_xy
        smooth_kp[:, :2] = (
            self.ema_alpha       * smooth_kp[:, :2]
            + (1.0 - self.ema_alpha) * state.prev_keypoints[:, :2]
        )

        # ── 상태 갱신 ──────────────────────────────────────────────────────
        state.prev_keypoints = smooth_kp.copy()

        return smooth_kp.astype(np.float32)