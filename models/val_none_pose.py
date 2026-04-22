from ultralytics import YOLO

# YOLO26 detection 모델 (pose 아님!)
model = YOLO("yolo26n.pt")

# 이미지 추론
results = model("person1.jpg", save=True)

# 결과에서 bounding box 출력
for r in results:
    boxes = r.boxes
    for box in boxes:
        xyxy = box.xyxy[0]  # [x1, y1, x2, y2]
        conf = box.conf[0]  # confidence
        cls = box.cls[0]    # class id

        print(f"BBOX: {xyxy}, conf: {conf}, class: {cls}")