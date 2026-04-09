# core/tracker.py

"""
ByteTrack 기반 다중 객체 추적기
================================
supervision 라이브러리의 sv.ByteTrack 을 래핑하여
PoseModel.infer() 반환값과 연동

파이프라인:
  PoseModel.infer()        →  persons (keypoints 포함, track_id 없음)
  Tracker.update()         →  track_ids (persons 와 같은 순서)
  PoseEstimator.update()   →  track_id 별 keypoint 평활화
  FallDetector.update()    →  track_id 별 낙상 판정

설치:
  pip install supervision

참고:
  supervision ByteTrack + YOLOv11-pose 공식 파이프라인:
  https://supervision.roboflow.com/latest/how_to/track_objects/
"""

import numpy as np
import supervision as sv


class Tracker:
    """
    sv.ByteTrack 래퍼
    PoseModel.infer() 의 persons 리스트를 받아
    각 사람의 track_id 를 반환

    ByteTrack 동작 원리:
      1. 고신뢰도 bbox → 기존 트랙과 IoU 매칭 (1차 매칭)
      2. 미매칭 트랙 + 저신뢰도 bbox → 2차 매칭 (가림 처리 핵심)
      3. 매칭 실패 트랙 → lost 상태로 track_buffer 프레임 유지
      4. track_buffer 초과 → 트랙 삭제

    사용 예:
        tracker   = Tracker(tracker_cfg)
        track_ids = tracker.update(persons)
    """

    def __init__(self, tracker_cfg: dict | None = None):
        """
        Args:
            tracker_cfg: model.yaml 의 tracker 섹션 dict
                         None 이면 기본값 사용

                track_activation_threshold : 트랙 활성화 신뢰도 임계값 (기본 0.25)
                    고신뢰도 bbox 판단 기준 — 1차 매칭에 사용
                    낮을수록 더 많은 bbox 가 1차 매칭에 참여
                    워터파크처럼 가림이 많은 환경에서는 낮게 설정 권장

                lost_track_buffer : lost 상태 유지 프레임 수 (기본 30)
                    트랙이 사라진 후 이 프레임 수 동안 재등장을 기다림
                    FPS 에 맞게 조정: 10 FPS × 3초 = 30 프레임

                minimum_matching_threshold : IoU 매칭 임계값 (기본 0.8)
                    높을수록 엄격하게 매칭 (ID 혼동 감소, 재연결 어려움)
                    낮을수록 느슨하게 매칭 (재연결 쉬움, ID 혼동 증가)

                minimum_consecutive_frames : 트랙 확정 최소 연속 프레임 수 (기본 1)
                    N 프레임 연속 감지된 경우에만 트랙 ID 부여
                    1 이상으로 설정하면 순간 오감지 방지 가능
        """
        cfg = tracker_cfg or {}

        self._tracker = sv.ByteTrack(
            track_activation_threshold = cfg.get("track_activation_threshold", 0.25),
            lost_track_buffer          = cfg.get("lost_track_buffer",          30),
            minimum_matching_threshold = cfg.get("minimum_matching_threshold", 0.8),
            minimum_consecutive_frames = cfg.get("minimum_consecutive_frames", 1),
        )

    # ─────────────────────────────────────────
    # 퍼블릭 메서드
    # ─────────────────────────────────────────

    def update(self, persons: list[dict]) -> list[int]:
        """
        persons 리스트를 받아 track_id 리스트를 반환

        Args:
            persons: PoseModel.infer() 반환값
                     [{"bbox": [x1,y1,x2,y2], "conf": float, "keypoints": ndarray(17,3)}, ...]

        Returns:
            persons 와 같은 순서의 track_id 리스트
            감지된 사람이 없으면 빈 리스트 반환

        주의:
            ByteTrack 은 bbox 기반으로 매칭하므로
            keypoints 는 매칭에 사용되지 않음
            track_id 는 persons 리스트와 같은 순서로 반환됨
        """
        if not persons:
            return []

        # ── persons → sv.Detections 변환 ──────────────────────────────────
        detections = self._to_sv_detections(persons)

        # ── ByteTrack 업데이트 ────────────────────────────────────────────
        tracked = self._tracker.update_with_detections(detections)

        # ── track_id 추출 및 persons 순서에 맞게 정렬 ─────────────────────
        track_ids = self._align_track_ids(persons, tracked)

        return track_ids

    def reset(self) -> None:
        """
        트래커 상태 초기화
        카메라 전환, 장면이 크게 바뀌는 경우 호출
        """
        self._tracker.reset()

    # ─────────────────────────────────────────
    # 내부 처리 메서드
    # ─────────────────────────────────────────

    @staticmethod
    def _to_sv_detections(persons: list[dict]) -> sv.Detections:
        """
        persons 리스트 → sv.Detections 변환

        sv.ByteTrack 은 sv.Detections 를 입력으로 받음
        bbox (xyxy) 와 confidence 만 필요

        Returns:
            sv.Detections (xyxy, confidence 포함)
        """
        bboxes = np.array(
            [p["bbox"] for p in persons],
            dtype=np.float32,
        )   # (N, 4)

        confs = np.array(
            [p["conf"] for p in persons],
            dtype=np.float32,
        )   # (N,)

        # class_id 는 모두 0 (person 단일 클래스)
        class_ids = np.zeros(len(persons), dtype=int)

        return sv.Detections(
            xyxy       = bboxes,
            confidence = confs,
            class_id   = class_ids,
        )

    @staticmethod
    def _align_track_ids(
        persons: list[dict],
        tracked: sv.Detections,
    ) -> list[int]:
        """
        ByteTrack 반환 Detections 의 track_id 를
        원본 persons 리스트의 순서에 맞게 정렬

        ByteTrack 은 내부적으로 일부 bbox 를 필터링하거나
        순서를 바꿀 수 있으므로 bbox IoU 로 재매칭

        매칭 실패한 persons (track_id 없음) 는 -1 로 처리

        Returns:
            persons 와 같은 순서의 track_id 리스트
            매칭 실패 시 해당 인덱스는 -1
        """
        track_ids = [-1] * len(persons)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return track_ids

        # 원본 bbox 와 tracked bbox 를 IoU 로 매칭
        for t_idx, t_bbox in enumerate(tracked.xyxy):
            best_iou   = 0.0
            best_p_idx = -1

            for p_idx, person in enumerate(persons):
                iou = _calc_iou(np.array(person["bbox"]), t_bbox)
                if iou > best_iou:
                    best_iou   = iou
                    best_p_idx = p_idx

            # IoU 0.5 이상인 경우에만 매칭 허용
            if best_iou >= 0.5 and best_p_idx >= 0:
                track_ids[best_p_idx] = int(tracked.tracker_id[t_idx])

        return track_ids


# ─────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────

def _calc_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    두 bbox (xyxy) 의 IoU 계산

    Args:
        box_a, box_b: [x1, y1, x2, y2]

    Returns:
        IoU (0.0 ~ 1.0)
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter   = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0