# main.py
import cv2
import yaml
import threading
from models.depth_model import DepthModel
from models.pose_model import PoseModel
from core.depth_estimator import DepthEstimator
from core.pose_estimator import PoseEstimator
from core.tracker import Tracker
from core.fall_detector import FallDetector
from core.impact_estimator import ImpactEstimator
from utils.visualizer import Visualizer
from utils.alert import Alert

def main():
    # 1. 설정 로드
    with open("config/camera.yaml") as f:
        cam_cfg = yaml.safe_load(f)
    with open("config/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("config/thresholds.yaml") as f:
        thr_cfg = yaml.safe_load(f)

    # 2. 모델 초기화
    depth_model   = DepthModel(model_cfg)
    pose_model    = PoseModel(model_cfg)

    # 3. 핵심 모듈 초기화
    depth_est  = DepthEstimator(depth_model, cam_cfg)
    pose_est   = PoseEstimator(pose_model)
    tracker    = Tracker()
    fall_det   = FallDetector(thr_cfg)
    impact_est = ImpactEstimator(thr_cfg)
    viz        = Visualizer()
    alert      = Alert(thr_cfg)

    # 4. CCTV 스트림 연결
    cap = cv2.VideoCapture(cam_cfg["stream_url"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 5. DA2 + YOLO 병렬 추론
        depth_map_ref = [None]
        persons_ref   = [None]

        def run_depth():
            depth_map_ref[0] = depth_est.infer(frame)
        def run_pose():
            persons_ref[0] = pose_est.infer(frame)

        t1 = threading.Thread(target=run_depth)
        t2 = threading.Thread(target=run_pose)
        t1.start(); t2.start()
        t1.join();  t2.join()

        depth_map = depth_map_ref[0]
        persons   = persons_ref[0]

        # 6. 추적 + 낙상 판정
        tracked = tracker.update(persons, frame)
        for person in tracked:
            kp_3d = depth_est.project_keypoints(person.keypoints, depth_map)
            is_fall, velocity = fall_det.update(person.id, kp_3d)

            if is_fall:
                danger, impulse = impact_est.estimate(velocity)
                alert.send(person.id, danger, impulse, frame)

        # 7. 디버그 시각화 (선택)
        out = viz.draw(frame, tracked, depth_map)
        cv2.imshow("Fall Detection", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()