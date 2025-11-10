"""
Kiểm tra nhanh môi trường cho Bước 2 (Tracking)
- Import boxmot, ultralytics, torch
- In version và tình trạng GPU
"""

import sys


def main():
    print("=" * 70)
    print("CHECK BOXMOT/ULTRALYTICS/TORCH ENV")
    print("=" * 70)

    # Ultralytics
    try:
        import ultralytics
        print(f"[✓] ultralytics: {ultralytics.__version__}")
    except Exception as e:
        print(f"[✗] ultralytics import failed: {e}")

    # BoxMOT
    try:
        import boxmot
        ver = getattr(boxmot, "__version__", "unknown")
        print(f"[✓] boxmot: {ver}")
    except Exception as e:
        print(f"[✗] boxmot import failed: {e}")

    # Torch & CUDA
    try:
        import torch
        print(f"[✓] torch: {torch.__version__}")
        if torch.cuda.is_available():
            print("[✓] CUDA available")
            print(f"    device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("[!] CUDA not available -> chạy CPU")
    except Exception as e:
        print(f"[✗] torch import failed: {e}")

    print("-" * 70)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")


if __name__ == "__main__":
    main()
