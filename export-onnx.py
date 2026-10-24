from ultralytics import YOLO

YOLO("models/best.pt").export(
    format="onnx",
    imgsz=640,
    opset=18,
    simplify=True,
    dynamic=False,
    half=True,
    batch=1,
    device="cpu",
)