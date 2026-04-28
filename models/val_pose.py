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

# 결과 저장용 리스트
results_table = []

for model_name in models:
    model = YOLO(model_name)

    model("image6.png", save=False, verbose=False)

    start_time = time.time()
    results = model("CCTV.mp4", conf=0.15, save=True, exist_ok=False)
    infer_time = time.time() - start_time

    persons = 0
    min_conf = "-"

    for r in results:
        if r.keypoints is None:
            continue

        xy   = r.keypoints.xy.cpu().numpy()
        conf = r.keypoints.conf.cpu().numpy()
        persons = len(xy)

        if persons > 0:
            min_conf = f"{conf[0].min():.3f}"

    results_table.append({
        "model":      model_name,
        "infer_time": infer_time,
        "persons":    persons,
        "min_conf":   min_conf,
    })

# ── 테이블 출력 ──────────────────────────────────────────
col_w = [20, 14, 10, 12]   # 각 컬럼 너비

header = (
    f"{'Model':<{col_w[0]}}"
    f"{'Infer(sec)':^{col_w[1]}}"
    f"{'Persons':^{col_w[2]}}"
    f"{'Min Conf':^{col_w[3]}}"
)
sep = "-" * sum(col_w)

print("\n" + sep)
print(header)
print(sep)

for row in results_table:
    print(
        f"{row['model']:<{col_w[0]}}"
        f"{row['infer_time']:^{col_w[1]}.3f}"
        f"{row['persons']:^{col_w[2]}}"
        f"{str(row['min_conf']):^{col_w[3]}}"
    )

print(sep)