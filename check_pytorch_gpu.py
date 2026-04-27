import sys
import time


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is not installed in this Python environment.")
        print(f"Python executable: {sys.executable}")
        return

    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("PyTorch cannot access the GPU in the current environment.")
        print("Common reasons:")
        print("- CPU-only PyTorch is installed")
        print("- NVIDIA driver is missing or outdated")
        print("- VS Code is using a different Python environment")
        return

    device = torch.device("cuda:0")
    print(f"Selected device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Tensor device: {c.device}")
    print(f"Matrix multiply test completed in {elapsed:.4f} seconds")
    print("PyTorch is using the GPU correctly.")


if __name__ == "__main__":
    main()
