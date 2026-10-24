import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorrt as trt
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
VIDEO = Path(r"C:\Users\M\Desktop\original_i.mp4")
OUT = Path(r"C:\Users\M\Desktop\detect")
PT_PATH = MODELS / "best.pt"
ONNX_PATH = MODELS / "best.onnx"
ENGINE_PATH = MODELS / "best.engine"

IMGSZ = 640
CONF = 0.25
IOU = 0.7
MAX_DET = 300
DEVICE = 0
WARMUP = 10

OUT.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(1)


def draw(frame, det, label, names):
    boxes, confs, classes, kpts = det
    out = frame.copy()
    for box, conf, cls, kp in zip(boxes, confs, classes, kpts):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{names.get(int(cls), cls)} {conf:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        pts = np.asarray(kp, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        for p in pts:
            cv2.circle(out, tuple(p[0]), 4, (0, 0, 255), -1)
    cv2.putText(out, f"{label} {len(boxes)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


class TRTPose:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.output_shape = tuple(self.engine.get_tensor_shape("output0"))
        self.lb = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)
        self.stream = torch.cuda.Stream()

    def infer(self, frame):
        params = self.lb.get_params({"img": frame})
        im = self.lb(image=frame)
        x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255

        with torch.cuda.stream(self.stream):
            inp = torch.from_numpy(x).cuda()
            out = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address("images", inp.data_ptr())
            self.context.set_tensor_address("output0", out.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        pred = out.cpu().numpy()[0].T
        boxes = pred[:, :4].copy()
        scores = pred[:, 4:10]
        kpts = pred[:, 10:18].reshape(-1, 4, 2)
        conf = scores.max(1)
        cls = scores.argmax(1)
        mask = conf >= CONF
        boxes, conf, cls, kpts = boxes[mask], conf[mask], cls[mask], kpts[mask]
        if not len(boxes):
            return boxes, conf, cls, kpts

        xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2,
                               boxes[:, :2] + boxes[:, 2:] / 2], 1)
        keep = nms(torch.from_numpy(xyxy).cuda() +
                   torch.from_numpy(cls).float().cuda()[:, None] * 7680,
                   torch.from_numpy(conf).cuda(), IOU)[:MAX_DET].cpu().numpy()
        xyxy, conf, cls, kpts = xyxy[keep], conf[keep], cls[keep], kpts[keep]

        r = params["ratio"]
        dw, dh = params["left"], params["top"]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r[0]
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r[1]
        kpts[:, :, 0] = (kpts[:, :, 0] - dw) / r[0]
        kpts[:, :, 1] = (kpts[:, :, 1] - dh) / r[1]
        return xyxy, conf, cls, kpts


def pred_to_det(res):
    boxes = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    kpts = (res.keypoints.xy.cpu().numpy() if res.keypoints is not None
            else np.zeros((0, 4, 2)))
    return boxes, conf, cls, kpts


def run_model(key, infer_fn, info, names):
    w, h, fps, _ = info
    cap = cv2.VideoCapture(str(VIDEO))
    writer = cv2.VideoWriter(str(OUT / f"{key}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    times, counts, confs = [], [], []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        det = infer_fn(frame)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        boxes, conf, _, _ = det
        if idx >= WARMUP:
            times.append(dt)
            counts.append(len(boxes))
            confs.extend(conf.tolist())
        writer.write(draw(frame, det, key.upper(), names))
        idx += 1

    cap.release()
    writer.release()
    arr = np.asarray(times)
    print(f"{key}: frames={idx} boxes={sum(counts)} "
          f"median={np.median(arr):.2f}ms p95={np.percentile(arr, 95):.2f}ms "
          f"conf={np.mean(confs) if confs else 0:.3f}")


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO}")
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 30,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

    pt = YOLO(str(PT_PATH), task="pose")
    onnx = YOLO(str(ONNX_PATH), task="pose")
    trt_model = TRTPose(ENGINE_PATH)
    names = pt.names

    for key, model in {"pt": pt, "onnx": onnx, "tensorrt": trt_model}.items():
        if key == "tensorrt":
            infer_fn = model.infer
        else:
            infer_fn = lambda frame, m=model: pred_to_det(
                m.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU,
                          max_det=MAX_DET, device=DEVICE, verbose=False)[0])
        run_model(key, infer_fn, info, names)


if __name__ == "__main__":
    main()
