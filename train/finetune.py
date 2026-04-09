# train/finetune.py

"""
YOLOv11-pose 파인튜닝 스크립트
================================
COCO pretrained YOLOv11-pose 를 워터파크 낙상 감지용으로 파인튜닝

사용법:
    python train/finetune.py                         # 기본 설정으로 실행
    python train/finetune.py --epochs 50             # 에폭 수 변경
    python train/finetune.py --resume runs/exp1      # 학습 재개
    python train/finetune.py --data data/custom.yaml # 다른 데이터셋 사용

데이터셋 준비:
    1. data/annotated/ 에 YOLO pose 형식으로 이미지 + 라벨 배치
    2. data/dataset.yaml 경로/클래스 설정 확인
    3. 이 스크립트 실행

결과:
    train/runs/pose/exp{N}/
        weights/best.pt   ← 추론에 사용할 최종 가중치
        weights/last.pt   ← 마지막 체크포인트
        results.csv       ← 에폭별 loss/metric 기록
        args.yaml         ← 학습에 사용된 전체 설정

학습 완료 후 model.yaml 의 pose.checkpoint 를 best.pt 경로로 변경:
    pose:
      checkpoint: "train/runs/pose/exp1/weights/best.pt"
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


# ─────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────

DEFAULTS = {
    # ── 모델 ──────────────────────────────
    # pretrained 가중치: 크기별로 n/s/m/l/x 선택 가능
    # 실시간 처리가 필요하므로 n(nano) 또는 s(small) 권장
    "model":     "yolo11n-pose.pt",

    # ── 데이터 ────────────────────────────
    "data":      "data/dataset.yaml",

    # ── 학습 하이퍼파라미터 ───────────────
    "epochs":    50,        # 에폭 수 (데이터셋 크기에 따라 조정)
    "patience":  10,        # Early stopping: N 에폭 동안 개선 없으면 중단
    "batch":     16,        # 배치 크기 (GPU 메모리에 맞게 조정, -1 이면 자동)
    "imgsz":     640,       # 입력 이미지 크기

    # ── 옵티마이저 ────────────────────────
    "optimizer": "AdamW",   # SGD / Adam / AdamW
    "lr0":       1e-4,      # 초기 학습률 (pretrained 파인튜닝은 낮게 설정)
    "lrf":       0.01,      # 최종 학습률 = lr0 × lrf
    "momentum":  0.937,
    "weight_decay": 5e-4,

    # ── 전이학습 설정 ─────────────────────
    # freeze: 백본 레이어를 고정하여 헤드만 학습
    # 데이터가 적을 때 유용 (과적합 방지)
    # None: 전체 학습, 10: 처음 10개 레이어 고정
    "freeze":    10,

    # ── 데이터 증강 ───────────────────────
    # 워터파크 환경 특성 반영
    "flipud":    0.0,       # 상하 반전 (낙상 감지에서는 비활성화)
    "fliplr":    0.5,       # 좌우 반전
    "hsv_h":     0.015,     # 색조 변화 (물 반사 대응)
    "hsv_s":     0.7,       # 채도 변화
    "hsv_v":     0.4,       # 밝기 변화 (실내/실외 조명 대응)
    "degrees":   5.0,       # 회전 (카메라 설치 각도 오차 대응)
    "scale":     0.5,       # 스케일 변화 (어린이/성인 혼재 대응)
    "mosaic":    1.0,       # 모자이크 증강
    "mixup":     0.1,       # MixUp 증강

    # ── 저장 설정 ─────────────────────────
    "project":   "train/runs",
    "name":      "pose",    # runs/pose/exp{N} 형태로 자동 증가
    "save":      True,
    "save_period": 10,      # N 에폭마다 체크포인트 저장

    # ── 기타 ──────────────────────────────
    "device":    "",        # "" = 자동 선택, "cpu", "0", "0,1" (GPU ID)
    "workers":   4,         # 데이터 로더 워커 수
    "verbose":   True,
    "seed":      42,        # 재현성
}


# ─────────────────────────────────────────
# 학습 실행
# ─────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    """파인튜닝 실행"""

    # ── 설정 병합 (기본값 + CLI 인자) ─────────────────────────────────────
    cfg = DEFAULTS.copy()
    if args.resume:
        cfg["resume"] = str(Path(args.resume) / "weights" / "last.pt")
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.freeze is not None:
        cfg["freeze"] = args.freeze
    if args.data:
        cfg["data"] = args.data
    if args.model:
        cfg["model"] = args.model
    if args.device:
        cfg["device"] = args.device
    if args.name:
        cfg["name"] = args.name

    # ── 데이터셋 yaml 존재 확인 ──────────────────────────────────────────
    data_path = Path(cfg["data"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"데이터셋 설정 파일을 찾을 수 없습니다: {data_path}\n"
            f"data/dataset.yaml 을 먼저 작성해 주세요."
        )

    # ── 데이터셋 정보 출력 ───────────────────────────────────────────────
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)
    print("\n[ 데이터셋 정보 ]")
    print(f"  경로  : {data_cfg.get('path', '-')}")
    print(f"  train : {data_cfg.get('train', '-')}")
    print(f"  val   : {data_cfg.get('val', '-')}")
    print(f"  클래스: {data_cfg.get('names', '-')}")
    print(f"  keypoint shape: {data_cfg.get('kpt_shape', '-')}\n")

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    # resume 이면 last.pt 에서 이어서 학습
    model_path = cfg.pop("resume", cfg.pop("model"))
    print(f"[ 모델 로드 ] {model_path}")
    model = YOLO(model_path)

    # ── 학습 실행 ─────────────────────────────────────────────────────────
    print("[ 파인튜닝 시작 ]")
    results = model.train(**cfg)

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print("\n[ 학습 완료 ]")
    print(f"  최고 가중치 : {best_path}")
    print(f"  결과 저장   : {results.save_dir}")
    print("\n[ 다음 단계 ]")
    print(f"  model.yaml 의 pose.checkpoint 를 아래 경로로 변경하세요:")
    print(f"  checkpoint: \"{best_path}\"")


# ─────────────────────────────────────────
# CLI 인터페이스
# ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv11-pose 워터파크 낙상 감지 파인튜닝",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model", type=str, default=None,
        help="pretrained 가중치 경로 또는 모델 ID (예: yolo11n-pose.pt)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="데이터셋 yaml 경로 (예: data/dataset.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="학습 에폭 수",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="배치 크기 (-1 이면 자동)",
    )
    parser.add_argument(
        "--freeze", type=int, default=None,
        help="고정할 레이어 수 (None = 전체 학습, 10 = 백본 고정)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="학습 재개할 runs 폴더 경로 (예: train/runs/pose/exp1)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="학습 장치 ('' = 자동, 'cpu', '0', '0,1')",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="실험 이름 (runs/pose/{name} 에 저장)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)