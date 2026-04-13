# core/fall_detector.py

"""
낙상 감지 알고리즘
==================
참고 논문:
  [논문 1] Tîrziu et al., "Real-Time Fall Monitoring for Seniors via YOLO and Voice Interaction",
           Future Internet, 2025, 17, 324.  (YOLOv11-pose 기반)
  [논문 2] Tîrziu et al., "Enhanced Fall Detection Using YOLOv7-W6-Pose for Real-Time Elderly
           Monitoring", Future Internet, 2024, 16, 472.  (YOLOv7-W6-Pose 기반)

위험도 판정은 core/impact_estimator.py 에서 담당
"""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from models.pose_model import PoseModel
from models.depth_model import DepthModel


# ─────────────────────────────────────────
# 열거형 정의
# ─────────────────────────────────────────

class PostureState(Enum):
    """
    자세 상태 4단계 분류 [논문 1]
    standing → leaning → fallen 순으로 전이
    """
    STANDING = "standing"   # 정상 기립
    SITTING  = "sitting"    # 착석
    LEANING  = "leaning"    # 상체 기울임 (낙상 전조)
    FALLEN   = "fallen"     # 낙상


# ─────────────────────────────────────────
# 단일 사람의 상태 추적 데이터
# ─────────────────────────────────────────

@dataclass
class PersonState:
    """
    트랙 ID별로 이전 프레임 정보를 저장
    위험도 판정에 필요한 head 속도 이력은
    ImpactEstimator 의 ImpactState 에서 관리
    """
    track_id: int

    # 이전 프레임 픽셀 좌표
    prev_hip:  np.ndarray | None = None
    prev_head: np.ndarray | None = None
    prev_cog:  np.ndarray | None = None

    # 낙상 후보 연속 프레임 카운터 (confirm_frames 도달 시 낙상 확정)
    candidate_frames: int = 0

    # 낙상 확정 시각 (cooldown 계산용) [논문 2]
    last_fall_time: float = 0.0

    # 현재 자세 상태
    posture: PostureState = PostureState.STANDING


# ─────────────────────────────────────────
# 낙상 감지 결과
# ─────────────────────────────────────────

@dataclass
class FallResult:
    """
    FallDetector.update() 반환값
    위험도(danger_level, impulse)는 ImpactEstimator.update() 로 별도 계산
    """
    track_id:      int
    fall_detected: bool
    posture:       PostureState

    # 디버그 / 시각화용 수치
    angle_ratio:   float = 0.0          # 상체 기울기 ratio [논문 1]
    hip_velocity:  float = 0.0          # hip 실제 속도 (m/s)
    cog_velocity:  float = 0.0          # CoG 실제 속도 (m/s)
    head_velocity: float = 0.0          # head 실제 속도 (m/s), ImpactEstimator로 전달
    bbox_ratio:    float = 0.0          # bounding box W/H [논문 2]
    l_factor:      float | None = None  # 체형 보정 계수 [논문 2], ImpactEstimator로 전달


# ─────────────────────────────────────────
# 낙상 감지기 메인 클래스
# ─────────────────────────────────────────

class FallDetector:
    """
    단일 프레임에서 감지된 모든 사람에 대해 낙상 여부를 판정
    위험도 판정은 ImpactEstimator 에 위임

    사용 예:
        detector = FallDetector(threshold_cfg, camera_cfg)
        results  = detector.update(persons, depth_map, dt=0.1)
    """

    # 각도 계산 분모 0 방지 상수 [논문 1]
    _EPSILON = 1e-5

    # 성인 기준 L_factor 픽셀값 (어깨-엉덩이 거리)
    # _classify_posture 의 NORM 값과 동일하게 유지
    _REFERENCE_L = 150.0

    def __init__(self, threshold_cfg: dict, camera_cfg: dict):
        """
        Args:
            threshold_cfg: thresholds.yaml 전체 dict
            camera_cfg:    camera.yaml 전체 dict
        """
        fd   = threshold_cfg["fall_detection"]
        intr = camera_cfg["intrinsics"]

        # ── 낙상 판정 임계값 ──────────────────────────────────────────────

        # 상체 기울기 ratio 임계값 [논문 1]
        # |Δy / (Δx + ε)| < 이 값이면 몸이 수평에 가까움
        self.body_angle_ratio_threshold: float = fd["body_angle_ratio_threshold"]

        # hip 실제 이동 속도 임계값 (m/s)
        self.hip_velocity_threshold: float = fd["hip_velocity_threshold"]

        # hip y좌표 임계값 (픽셀) — 화면 하단에 있어야 낙상
        self.hip_y_threshold: float = fd["hip_y_threshold"]

        # |y_hips - y_feet| 임계값 (픽셀) [논문 1]
        # 이 값 미만이면 몸이 수평으로 누운 상태
        self.feet_hip_y_diff_threshold: float = fd["feet_hip_y_diff_threshold"]

        # bbox W/H 임계값 [논문 2]
        # 이 값 이상이면 몸이 가로로 누운 상태
        self.bbox_ratio_threshold: float = fd["bbox_ratio_threshold"]

        # CoG 수직 이동 속도 임계값 (m/s) [논문 1]
        self.cog_velocity_threshold: float = fd["cog_velocity_threshold"]

        # 연속 N프레임 낙상 후보 충족 시 확정
        self.confirm_frames: int = fd["confirm_frames"]

        # 낙상 후 재판정 억제 시간 (초) [논문 2]
        self.cooldown_sec: float = fd["cooldown_sec"]

        # ── 카메라 내부 파라미터 ──────────────────────────────────────────
        self.fx: float = intr["fx"]
        self.fy: float = intr["fy"]

        # ── 사람별 상태 저장 ─────────────────────────────────────────────
        # {track_id: PersonState}
        self._states: dict[int, PersonState] = {}

    # ─────────────────────────────────────────
    # 퍼블릭 메서드
    # ─────────────────────────────────────────

    def update(
        self,
        persons:   list[dict],
        depth_map: np.ndarray,
        dt:        float,
        track_ids: list[int] | None = None,
    ) -> list[FallResult]:
        """
        프레임 단위로 낙상 여부를 판정

        Args:
            persons:   PoseModel.infer() 반환값
                       [{"bbox": [...], "conf": float, "keypoints": ndarray(17,3)}, ...]
            depth_map: DepthModel.infer() 반환값 (H, W), float32, 단위 m
            dt:        직전 프레임과의 시간 간격 (초), 예: 0.1 (10 FPS)
            track_ids: persons와 같은 순서의 트랙 ID 리스트
                       None이면 persons 리스트의 인덱스를 ID로 사용

        Returns:
            persons와 같은 순서의 FallResult 리스트
        """
        if track_ids is None:
            track_ids = list(range(len(persons)))

        results = []
        for person, tid in zip(persons, track_ids):
            result = self._process_person(person, tid, depth_map, dt)
            results.append(result)

        # 이번 프레임에 없는 트랙 ID의 상태 정리
        active_ids = set(track_ids)
        lost_ids   = [tid for tid in self._states if tid not in active_ids]
        for tid in lost_ids:
            del self._states[tid]

        return results

    # ─────────────────────────────────────────
    # 내부 처리 메서드
    # ─────────────────────────────────────────

    def _get_or_create_state(self, track_id: int) -> PersonState:
        if track_id not in self._states:
            self._states[track_id] = PersonState(track_id=track_id)
        return self._states[track_id]

    def _process_person(
        self,
        person:    dict,
        track_id:  int,
        depth_map: np.ndarray,
        dt:        float,
    ) -> FallResult:
        """단일 사람에 대한 낙상 판정 전체 파이프라인"""
        kp    = person["keypoints"]   # (17, 3)
        bbox  = person["bbox"]        # [x1, y1, x2, y2]
        state = self._get_or_create_state(track_id)

        # ── 키포인트 추출 ──────────────────────────────────────────────────
        hip      = PoseModel.hip_center(kp)
        head     = PoseModel.head_center(kp)
        shoulder = PoseModel.shoulder_center(kp)
        feet     = PoseModel.feet_center(kp)
        cog      = PoseModel.cog(kp)
        l_factor = PoseModel.l_factor(kp)

        # ── 깊이 조회 ──────────────────────────────────────────────────────
        hip_depth  = DepthModel.get_depth_at_keypoint(depth_map, hip)
        head_depth = DepthModel.get_depth_at_keypoint(depth_map, head)
        cog_depth  = DepthModel.get_depth_at_keypoint(depth_map, cog)

        # ── 실제 속도 계산 (m/s) ───────────────────────────────────────────
        hip_velocity, cog_velocity, head_velocity = self._calc_velocities(
            state, hip, head, cog,
            hip_depth, head_depth, cog_depth,
            dt,
        )

        # ── 각도 계산 [논문 1] ─────────────────────────────────────────────
        angle_ratio = self._calc_angle_ratio(shoulder, hip)

        # ── bbox W/H 비율 [논문 2] ────────────────────────────────────────
        bbox_ratio = self._calc_bbox_ratio(bbox)

        # ── 자세 분류 + 낙상 후보 판정 (통합) ───────────────────────────
        posture, is_fall_candidate = self._assess(
            angle_ratio, hip_velocity, cog_velocity,
            hip, shoulder, feet, bbox_ratio, l_factor,
        )

        # ── N프레임 연속 확정 ──────────────────────────────────────────────
        fall_confirmed = self._confirm_fall(state, is_fall_candidate)

        # ── cooldown 확인 [논문 2] ────────────────────────────────────────
        now = time.time()
        in_cooldown   = (now - state.last_fall_time) < self.cooldown_sec
        fall_detected = fall_confirmed and not in_cooldown

        if fall_detected:
            state.last_fall_time = now
            state.posture        = PostureState.FALLEN
        else:
            state.posture = posture

        # ── 상태 갱신 ──────────────────────────────────────────────────────
        state.prev_hip  = hip
        state.prev_head = head
        state.prev_cog  = cog

        return FallResult(
            track_id      = track_id,
            fall_detected = fall_detected,
            posture       = state.posture,
            angle_ratio   = angle_ratio,
            hip_velocity  = hip_velocity,
            cog_velocity  = cog_velocity,
            head_velocity = head_velocity,   # ImpactEstimator 로 전달
            bbox_ratio    = bbox_ratio,
            l_factor      = l_factor,        # ImpactEstimator 로 전달
        )

    # ─────────────────────────────────────────
    # 계산 헬퍼 메서드
    # ─────────────────────────────────────────

    def _calc_velocities(
        self,
        state:      PersonState,
        hip:        np.ndarray | None,
        head:       np.ndarray | None,
        cog:        np.ndarray | None,
        hip_depth:  float | None,
        head_depth: float | None,
        cog_depth:  float | None,
        dt:         float,
    ) -> tuple[float, float, float]:
        """
        hip / head / CoG 의 실제 속도(m/s)를 계산
        핀홀 카메라 역투영: ΔX_real = Δx_px × depth / fx

        Returns:
            (hip_velocity, cog_velocity, head_velocity)  단위: m/s
        """
        def _velocity(
            cur:   np.ndarray | None,
            prev:  np.ndarray | None,
            depth: float | None,
        ) -> float:
            if cur is None or prev is None or depth is None or depth <= 0:
                return 0.0
            dx = float(cur[0] - prev[0])
            dy = float(cur[1] - prev[1])
            _, _, speed = DepthModel.get_real_velocity(
                dx, dy, depth, self.fx, self.fy, dt,
            )
            return speed

        hip_v  = _velocity(hip,  state.prev_hip,  hip_depth)
        head_v = _velocity(head, state.prev_head, head_depth)
        cog_v  = _velocity(cog,  state.prev_cog,  cog_depth)

        return hip_v, cog_v, head_v

    def _calc_angle_ratio(
        self,
        shoulder: np.ndarray | None,
        hip:      np.ndarray | None,
    ) -> float:
        """
        상체 기울기 ratio 계산 [논문 1]
        angle_ratio = |Δy / (Δx + ε)|
        수직(기립) 시 ratio 크고, 수평(낙상) 시 ratio 작음
        """
        if shoulder is None or hip is None:
            return 999.0   # 감지 실패 → 낙상 조건 불충족 처리

        dy = float(hip[1] - shoulder[1])
        dx = float(hip[0] - shoulder[0])
        return abs(dy / (abs(dx) + self._EPSILON))

    def _calc_bbox_ratio(self, bbox: list) -> float:
        """
        bounding box W/H 비율 계산 [논문 2]
        1.0 이상이면 가로가 세로보다 긴 상태 (누운 상태)
        """
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        return w / h

    def _assess(
        self,
        angle_ratio:  float,
        hip_velocity: float,
        cog_velocity: float,
        hip:          np.ndarray | None,
        shoulder:     np.ndarray | None,
        feet:         np.ndarray | None,
        bbox_ratio:   float,
        l_factor:     float | None,
    ) -> tuple[PostureState, bool]:
        """
        자세 분류 + 낙상 후보 판정을 한 번에 수행

        _classify_posture() 와 _is_fall_candidate() 를 통합
        두 함수가 공통으로 사용하는 값(hip_y, feet_y, margin 등)을
        한 번만 계산하여 중복 연산 제거

        Returns:
            (PostureState, is_fall_candidate)
        """
        # ── 공통 전처리 ───────────────────────────────────────────────────

        # hip 없으면 판정 불가
        if hip is None:
            return PostureState.STANDING, False

        hip_y  = float(hip[1])
        feet_y = float(feet[1]) if feet is not None else None

        # L_factor 체형 보정 [논문 2]
        # 성인 기준 REFERENCE_L ≈ 150px
        lf    = l_factor if (l_factor and l_factor > 0) else self._REFERENCE_L
        scale = lf / self._REFERENCE_L

        # 자세 분류용 마진 (픽셀, 논문 1 고정값을 체형에 맞게 보정)
        margin_fall  = 20.0 * scale   # 낙상 판정 y 마진
        margin_horiz = 30.0 * scale   # 수평 판정 y 마진

        # 속도 임계값 보정 (어린이 → 임계값 낮아짐) [논문 2]
        adj_hip_vel_threshold = self.hip_velocity_threshold * scale
        adj_cog_vel_threshold = self.cog_velocity_threshold * scale

        # 공통 조건 계산
        shoulder_y = float(shoulder[1]) if shoulder is not None else None

        hip_above_shoulder = (
            shoulder_y is not None
            and hip_y < shoulder_y - margin_fall
        )
        body_horizontal = (
            feet_y is not None
            and abs(hip_y - feet_y) < margin_horiz
        )

        # ── 자세 분류 [논문 1] ────────────────────────────────────────────
        if shoulder is None:
            posture = PostureState.STANDING
        elif angle_ratio < 0.5 and hip_above_shoulder and body_horizontal:
            posture = PostureState.FALLEN
        elif angle_ratio > 0.7 and hip_y > (shoulder_y or 0):
            posture = PostureState.STANDING
        elif angle_ratio > 0.6 and hip_y > (shoulder_y or 0):
            posture = PostureState.SITTING
        elif 0.3 < angle_ratio < 0.6 and feet_y is not None and hip_y > feet_y - margin_horiz:
            posture = PostureState.LEANING
        else:
            posture = PostureState.STANDING

        # ── 낙상 후보 판정 ────────────────────────────────────────────────
        # 핵심 조건 [논문 1 + 기존 의사 코드]
        cond_angle    = angle_ratio < self.body_angle_ratio_threshold          # [A]
        cond_hip_vel  = hip_velocity > adj_hip_vel_threshold                   # [B]
        cond_hip_y    = hip_y > self.hip_y_threshold                           # [C]
        cond_feet_hip = body_horizontal                                        # [D] 공통값 재사용

        core_condition = cond_angle and cond_hip_vel and cond_hip_y and cond_feet_hip

        # 보조 조건 [논문 2]
        cond_bbox    = bbox_ratio > self.bbox_ratio_threshold                  # [E]
        cond_cog_vel = cog_velocity > adj_cog_vel_threshold                    # [F]

        aux_condition = cond_bbox and (cond_hip_vel or cond_cog_vel)

        is_candidate = core_condition or aux_condition

        return posture, is_candidate

    def _confirm_fall(self, state: PersonState, is_candidate: bool) -> bool:
        """
        N프레임 연속 낙상 후보 충족 시 낙상으로 확정
        단일 프레임 오감지 방지
        """
        if is_candidate:
            state.candidate_frames += 1
        else:
            state.candidate_frames = 0

        return state.candidate_frames >= self.confirm_frames