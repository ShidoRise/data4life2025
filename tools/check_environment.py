"""
SCRIPT KIỂM TRA MÔI TRƯỜNG VÀ CÀI ĐẶT
====================================

Script này sẽ:
1. Kiểm tra Python environment
2. Kiểm tra các thư viện cần thiết
3. Kiểm tra CUDA/GPU (nếu có)
4. Download model YOLOv8n nếu chưa có
5. Chạy test detection đơn giản

"""

import sys
import subprocess
from pathlib import Path


def check_environment():
    """Kiểm tra môi trường Python"""
    print("=" * 70)
    print("KIỂM TRA MÔI TRƯỜNG")
    print("=" * 70)
    
    print(f"\n1. Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    
    # Kiểm tra conda environment
    conda_env = sys.prefix
    print(f"   Conda environment: {conda_env}")
    
    return True


def check_libraries():
    """Kiểm tra các thư viện cần thiết"""
    print("\n2. Kiểm tra thư viện:")
    
    libraries = {
        'ultralytics': 'YOLO framework',
        'boxmot': 'Multi-object tracking',
        'cv2': 'OpenCV - Computer vision',
        'torch': 'PyTorch - Deep learning',
        'numpy': 'Numerical computing',
        'pathlib': 'Path handling'
    }
    
    all_ok = True
    
    for lib, description in libraries.items():
        try:
            if lib == 'cv2':
                import cv2
                version = cv2.__version__
            elif lib == 'torch':
                import torch
                version = torch.__version__
            elif lib == 'ultralytics':
                import ultralytics
                version = ultralytics.__version__
            elif lib == 'boxmot':
                import boxmot
                version = getattr(boxmot, '__version__', 'unknown')
            elif lib == 'numpy':
                import numpy
                version = numpy.__version__
            elif lib == 'pathlib':
                import pathlib
                version = 'built-in'
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'unknown')
            
            status = "✓"
            print(f"   [{status}] {lib:20s} ({version:15s}) - {description}")
        except ImportError as e:
            all_ok = False
            status = "✗"
            print(f"   [{status}] {lib:20s} - CHƯA CÀI ĐẶT - {description}")
            print(f"       → Cài đặt: pip install {lib}")
    
    return all_ok


def check_gpu():
    """Kiểm tra GPU/CUDA"""
    print("\n3. Kiểm tra GPU:")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   [✓] CUDA available: {torch.cuda.is_available()}")
            print(f"   [✓] CUDA version: {torch.version.cuda}")
            print(f"   [✓] GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   [✓] GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   [✓] Current device: {torch.cuda.current_device()}")
            return True
        else:
            print("   [!] CUDA không khả dụng - Sẽ sử dụng CPU")
            print("   [!] Hiệu năng sẽ chậm hơn, nhưng vẫn hoạt động")
            return False
    except ImportError:
        print("   [✗] PyTorch chưa được cài đặt")
        return False


def check_models():
    """Kiểm tra model files"""
    print("\n4. Kiểm tra model files:")
    
    models = {
        'yolov8n.pt': 'YOLOv8 nano model',
        'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth': 'OSNet ReID model'
    }
    
    all_ok = True
    
    for model_file, description in models.items():
        model_path = Path(model_file)
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   [✓] {model_file}")
            print(f"       → {description} ({size_mb:.2f} MB)")
        else:
            print(f"   [✗] {model_file} - CHƯA CÓ")
            print(f"       → {description}")
            
            if model_file.startswith('yolov8'):
                print(f"       → Sẽ tự động download khi chạy lần đầu")
            else:
                all_ok = False
    
    return all_ok


def download_yolo_model():
    """Download YOLOv8n model nếu chưa có"""
    model_path = Path('yolov8n.pt')
    
    if model_path.exists():
        print("\n5. [✓] YOLOv8n model đã có sẵn")
        return True
    
    print("\n5. Đang download YOLOv8n model...")
    
    try:
        from ultralytics import YOLO
        
        # Load model sẽ tự động download
        model = YOLO('yolov8n.pt')
        print("   [✓] Download thành công!")
        return True
    except Exception as e:
        print(f"   [✗] Lỗi khi download: {e}")
        print("   → Vui lòng download thủ công từ:")
        print("      https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        return False


def test_detection():
    """Test detection với một ảnh mẫu"""
    print("\n6. Test detection:")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Load model
        model = YOLO('yolov8n.pt')
        print("   [✓] Model loaded successfully")
        
        # Tạo ảnh test (random)
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model.predict(test_image, verbose=False)
        print("   [✓] Inference test passed")
        print(f"   [✓] Model ready to use!")
        
        return True
    except Exception as e:
        print(f"   [✗] Test failed: {e}")
        return False


def print_summary(checks):
    """In tổng kết"""
    print("\n" + "=" * 70)
    print("TÓM TẮT KIỂM TRA")
    print("=" * 70)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   [{status}] {check_name}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n[SUCCESS] Tất cả kiểm tra đều PASS!")
        print("[INFO] Môi trường đã sẵn sàng cho Bước 1: Object Detection")
        print("\nChạy lệnh sau để bắt đầu:")
        print("  python step1_object_detection.py")
    else:
        print("\n[WARNING] Một số kiểm tra FAIL!")
        print("[INFO] Vui lòng cài đặt các thư viện còn thiếu:")
        print("  pip install ultralytics opencv-python torch")
    
    print()


def main():
    """Hàm chính"""
    print("\n" + "█" * 70)
    print("KIỂM TRA MÔI TRƯỜNG - MTMC PIPELINE BƯỚC 1")
    print("█" * 70)
    
    checks = {}
    
    # Chạy các kiểm tra
    checks['Environment'] = check_environment()
    checks['Libraries'] = check_libraries()
    checks['GPU/CUDA'] = check_gpu()
    checks['Model Files'] = check_models()
    checks['YOLO Download'] = download_yolo_model()
    checks['Detection Test'] = test_detection()
    
    # Tổng kết
    print_summary(checks)


if __name__ == '__main__':
    main()
