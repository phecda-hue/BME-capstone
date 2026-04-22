from ultralytics import YOLO
import time

# 테스트할 모델 리스트
models = [
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt"
]

image_path = "person.jpg"

for model_name in models:
    print("\n" + "="*50)
    print(f"Testing model: {model_name}")
    print("="*50)

    # 1. 모델 로드
    model = YOLO(model_name)

    # 2. 추론 시간 측정
    start_time = time.time()

    results = model(
        "image4.png",
        save=True,
        exist_ok=False
    )

    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference Time: {inference_time:.3f} sec")

    # 3. 결과 분석
    for r in results:
        if r.keypoints is None:
            print("No keypoints detected.")
            continue

        xy = r.keypoints.xy.cpu().numpy()
        conf = r.keypoints.conf.cpu().numpy()

        print(f"Detected persons: {len(xy)}")

        # 첫 번째 사람 기준 간단 비교
        if len(xy) > 0:
            min_conf = conf[0].min()
            print(f"Minimum keypoint confidence (person 0): {min_conf:.3f}")