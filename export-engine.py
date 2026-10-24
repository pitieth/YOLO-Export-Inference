from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
TRTEXEC = Path(r"G:\TensorRT-11.4.0.106\bin\trtexec.exe")


def build(onnx_path: Path, engine_path: Path):
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    cmd = [
        str(TRTEXEC),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:1G",
        "--skipInference",
    ]
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"built: {engine_path}", flush=True)


if __name__ == "__main__":
    build(MODELS / "best.onnx", MODELS / "best_fp32.engine")
    build(MODELS / "best_half.onnx", MODELS / "best_fp16.engine")
