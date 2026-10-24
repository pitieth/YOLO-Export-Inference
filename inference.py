import logging
import os
import warnings
from pathlib import Path

os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import tensorrt as trt
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER
import onnxruntime as ort

ort.set_default_logger_severity(3)
LOGGER.setLevel(logging.ERROR)
LOGGER.propagate = False

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SRC = Path(os.getenv("SOURCE", r"C:\Users\M\Desktop\original_i.mp4"))
OUT = Path(r"C:\Users\M\Desktop\detect\inference")
LIMIT = int(os.getenv("MAX_FRAMES", "0"))
NAMES = ["B1", "B2", "B3", "R1", "R2", "R3"]
IMGSZ, CONF, IOU, MAX_DET, DEVICE = 640, 0.25, 0.7, 300, 0


def draw(frame, det, label):
    boxes, confs, classes, kpts = det
    out = frame.copy()
    for box, conf, cls, kp in zip(boxes, confs, classes, kpts):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{NAMES[int(cls)]} {conf:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        pts = np.asarray(kp, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        for p in pts:
            cv2.circle(out, tuple(p[0]), 4, (0, 0, 255), -1)
    cv2.putText(out, f"{label} {len(boxes)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


class TRT:
    def __init__(self, path):
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
            Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.out_shape = tuple(self.engine.get_tensor_shape("output0"))
        self.lb = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)
        self.stream = torch.cuda.Stream()

    def __call__(self, frame):
        p = self.lb.get_params({"img": frame})
        im = self.lb(image=frame)
        x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        with torch.cuda.stream(self.stream):
            inp = torch.from_numpy(x).cuda()
            out = torch.empty(self.out_shape, device="cuda")
            self.ctx.set_tensor_address("images", inp.data_ptr())
            self.ctx.set_tensor_address("output0", out.data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        pred = out.cpu().numpy()[0].T
        boxes = pred[:, :4].copy()
        scores = pred[:, 4:10]
        kpts = pred[:, 10:18].reshape(-1, 4, 2)
        conf = scores.max(1)
        cls = scores.argmax(1)
        m = conf >= CONF
        boxes, conf, cls, kpts = boxes[m], conf[m], cls[m], kpts[m]
        if not len(boxes):
            return boxes, conf, cls, kpts

        xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2,
                               boxes[:, :2] + boxes[:, 2:] / 2], 1)
        keep = nms(torch.from_numpy(xyxy).cuda() +
                   torch.from_numpy(cls).float().cuda()[:, None] * 7680,
                   torch.from_numpy(conf).cuda(), IOU)[:MAX_DET].cpu().numpy()
        xyxy, conf, cls, kpts = xyxy[keep], conf[keep], cls[keep], kpts[keep]

        r = p["ratio"]
        dw, dh = p["left"], p["top"]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r[0]
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r[1]
        kpts[:, :, 0] = (kpts[:, :, 0] - dw) / r[0]
        kpts[:, :, 1] = (kpts[:, :, 1] - dh) / r[1]
        return xyxy, conf, cls, kpts


def infer(key, model, frame):
    if key == "tensorrt":
        return model(frame)
    r = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU,
                      max_det=MAX_DET, device=DEVICE, verbose=False)[0]
    kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 4, 2))
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), kpts


def main():
    models = {"pt": YOLO(str(MODELS / "best.pt"), task="pose"),
              "onnx": YOLO(str(MODELS / "best.onnx"), task="pose"),
              "tensorrt": TRT(MODELS / "best.engine")}
    OUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(SRC))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    for key, model in models.items():
        cap = cv2.VideoCapture(str(SRC))
        writer = cv2.VideoWriter(str(OUT / f"{SRC.stem}_{key}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(draw(frame, infer(key, model, frame), key.upper()))
            n += 1
            if LIMIT and n >= LIMIT:
                break
        cap.release()
        writer.release()

    print(f"saved {len(models)} file(s) -> {OUT}")


if __name__ == "__main__":
    main()
