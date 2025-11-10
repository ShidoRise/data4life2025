"""
Script to diagnose and fix torchreid issues
"""

print("=" * 70)
print("DIAGNOSING TORCHREID")
print("=" * 70)

# Test 1: Can we import torchreid?
print("\n[TEST 1] Importing torchreid...")
try:
    import torchreid
    print("✓ torchreid imported successfully")
    print(f"  Version: {torchreid.__version__ if hasattr(torchreid, '__version__') else 'unknown'}")
    print(f"  Location: {torchreid.__file__}")
except Exception as e:
    print(f"✗ Failed to import torchreid: {e}")
    import sys
    sys.exit(1)

# Test 2: Can we build OSNet?
print("\n[TEST 2] Building OSNet model...")
try:
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        loss='softmax',
        pretrained=False
    )
    print(f"✓ OSNet model built successfully")
    print(f"  Model type: {type(model)}")
    print(f"  Model name: {model.__class__.__name__}")
except Exception as e:
    print(f"✗ Failed to build OSNet: {e}")
    print(f"  Exception type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
    print("\n[DIAGNOSIS]")
    print("This error is likely due to missing dependencies.")
    print("\nTo fix, run:")
    print("  pip install tensorboard")
    print("  pip install tb-nightly")
    import sys
    sys.exit(1)

# Test 3: Can we load checkpoint?
print("\n[TEST 3] Loading checkpoint...")
import torch
from pathlib import Path

weights_file = "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

if not Path(weights_file).exists():
    print(f"✗ Weights file not found: {weights_file}")
else:
    try:
        checkpoint = torch.load(weights_file, map_location='cpu')
        print(f"✓ Checkpoint loaded")
        
        if isinstance(checkpoint, dict):
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"  State dict has {len(state_dict)} parameters")
        
        # Test loading into model
        print("\n[TEST 4] Loading weights into model...")
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ Weights loaded into model")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        if len(unexpected) > 0:
            print(f"\n  First few unexpected keys:")
            for key in list(unexpected)[:5]:
                print(f"    - {key}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
print("\nIf all tests passed, the issue is elsewhere.")
print("If TEST 2 failed with tensorboard error, run:")
print("  pip install tensorboard")
