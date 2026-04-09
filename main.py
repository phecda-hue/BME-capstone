# main.py

import time
import threading

import cv2
import yaml

from models.depth_model   import DepthModel
from models.pose_model    import PoseModel
from core.depth_estimator import DepthEstimator
from core.pose_estimator  import PoseEstimator
from core.fall_detector   import FallDetector
from core.impact_estimator import ImpactEstimator
from core.tracker          import Tracker
from utils.visualizer     import Visualizer
from utils.alert          import Alert


def main():
    # ── 1. 설정 로드 ──────────────────────────────────────────────────────
    with open("config/camera.yaml") as f:
        cam_cfg = yaml.safe_load(f)
    with open("config/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("config/thresholds.yaml") as f:
        thr_cfg = yaml.safe_load(f)

    # ── 2. 모델 초기화 ────────────────────────────────────────────────────
    # 추론 담당: DepthModel, PoseModel
    depth_model = DepthModel(model_cfg)
    pose_model  = PoseModel(model_cfg)

    # ── 3. 핵심 모듈 초기화 ──────────────────────────────────────────────
    # 후처리 담당: DepthEstimator, PoseEstimator
    depth_est  = DepthEstimator(model_cfg["depth_estimator"])
    pose_est   = PoseEstimator(model_cfg["pose_estimator"])

    # 낙상 판정 담당: FallDetector, ImpactEstimator
    fall_det   = FallDetector(thr_cfg, cam_cfg)
    impact_est = ImpactEstimator(thr_cfg)

    # 시각화 / 알림
    viz   = Visualizer(thr_cfg)
    alert = Alert(thr_cfg)

    # 트래커 (라이브러리 결정 후 활성화)
    # tracker = Tracker()

    # ── 4. 카메라 스트림 연결 ─────────────────────────────────────────────
    # camera.yaml 구조: cameras[0].stream_url
    stream_url = cam_cfg["cameras"][0]["stream_url"]
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        raise RuntimeError(f"카메라 스트림 연결 실패: {stream_url}")

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── 5. dt 계산 (프레임 간격, 초) ──────────────────────────────────
        now = time.time()
        dt  = now - prev_time
        dt  = max(dt, 1e-3)   # 0 나눗셈 방지
        prev_time = now

        # ── 6. 추론 (DA2 + YOLO 병렬 실행) ───────────────────────────────
        # 주의: 두 모델이 같은 GPU를 사용할 경우 GIL 및 CUDA 컨텍스트로 인해
        #       실제 병렬 실행이 제한될 수 있음
        # CPU 모델(depth: CPU, pose: GPU) 조합에서 효과적
        depth_result = [None]
        pose_result  = [None]

        def run_depth():
            raw_depth         = depth_model.infer(frame)
            depth_result[0]   = depth_est.update(raw_depth)   # EMA 평활화

        def run_pose():
            raw_persons      = pose_model.infer(frame)
            pose_result[0]   = raw_persons   # 평활화는 track_ids 확정 후 적용

        t1 = threading.Thread(target=run_depth)
        t2 = threading.Thread(target=run_pose)
        t1.start(); t2.start()
        t1.join();  t2.join()

        depth_map  = depth_result[0]
        raw_persons = pose_result[0]

        if depth_map is None or raw_persons is None:
            continue

        # ── 7. 트래킹 (트래커 결정 전: 인덱스를 임시 ID로 사용) ───────────
        # tracker 활성화 후 아래 두 줄을 교체
        # tracked   = tracker.update(raw_persons, frame)
        # track_ids = [t.id for t in tracked]
        track_ids = list(range(len(raw_persons)))

        # ── 8. keypoint 후처리 (EMA 평활화 + 신뢰도 보간) ────────────────
        smooth_persons = pose_est.update(raw_persons, track_ids)

        # ── 9. 낙상 판정 ──────────────────────────────────────────────────
        fall_results   = fall_det.update(smooth_persons, depth_map, dt, track_ids)
        impact_results = impact_est.update(fall_results, smooth_persons, depth_map)

        # ── 10. 알림 (낙상 감지 시) ───────────────────────────────────────
        for fall, impact in zip(fall_results, impact_results):
            if fall.fall_detected:
                # 비동기 전송으로 낙상 감지 루프 블로킹 방지 [논문 2]
                threading.Thread(
                    target=alert.send,
                    args=(fall, impact, frame),
                    daemon=True,
                ).start()

        # ── 11. 시각화 ────────────────────────────────────────────────────
        out = viz.draw(frame, smooth_persons, depth_map, fall_results, impact_results)
        cv2.imshow("Fall Detection", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()