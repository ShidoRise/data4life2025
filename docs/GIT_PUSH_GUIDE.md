# Git Push Guide - Hướng dẫn đẩy code lên GitHub

## 📋 Checklist trước khi push

- [x] README.md đã được tạo
- [x] requirements.txt đã có đầy đủ dependencies
- [x] .gitignore đã loại trừ files không cần thiết
- [x] LICENSE đã được thêm (MIT)
- [x] Code đã được test và chạy được
- [ ] Tạo repository trên GitHub
- [ ] Push code lên

## 🚀 Các bước thực hiện

### Bước 1: Initialize Git Repository (nếu chưa có)

```bash
cd d:\PYTHON\boxmot-test
git init
```

### Bước 2: Kiểm tra status

```bash
git status
```

Bạn sẽ thấy các files:
- ✅ Được track (màu xanh): .py files, .md files, .yaml configs
- ❌ Bị ignore (không hiện): *.pth, *.pt, *.mp4, runs/, *.pkl

### Bước 3: Add all files

```bash
git add .
```

**Lưu ý:** File `.gitignore` sẽ tự động loại trừ:
- Model weights: `*.pth`, `*.pt`, `*.onnx`
- Videos: `*.mp4`, `*.avi`
- Output directories: `runs/`, `outputs/`
- Pickle files: `*.pkl`
- Python cache: `__pycache__/`

### Bước 4: Xem files sẽ được commit

```bash
git status
```

Đảm bảo các files quan trọng được track:
```
✓ step1_object_detection.py
✓ step2_tracking.py
✓ step3_reid_extraction.py
✓ step4_inter_camera_association.py
✓ run_full_mtmc_pipeline.py
✓ botsort_config.yaml
✓ requirements.txt
✓ README.md
✓ SETUP_GUIDE.md
✓ STEP1-4_GUIDE.md
✓ .gitignore
✓ LICENSE
```

### Bước 5: Commit changes

```bash
git commit -m "Initial commit: Complete MTMC tracking pipeline

- Implemented 4-step MTMC pipeline (Detection, Tracking, Re-ID, Association)
- Added BoT-SORT tracking with Re-ID features
- OSNet feature extraction with memory optimization
- Hungarian and DBSCAN association methods
- Full automation scripts for multi-camera processing
- Comprehensive documentation and setup guides
"
```

### Bước 6: Tạo repository trên GitHub

1. Mở trình duyệt: https://github.com/new
2. Repository name: `mtmc-tracking` (hoặc tên bạn muốn)
3. Description: `Multi-Target Multi-Camera Tracking Pipeline with BoT-SORT and OSNet Re-ID`
4. Chọn: **Public** (hoặc Private nếu muốn)
5. **KHÔNG** chọn: "Initialize with README" (vì đã có sẵn)
6. Click **Create repository**

### Bước 7: Connect local repo với GitHub

GitHub sẽ hiển thị commands, copy và chạy:

```bash
git remote add origin https://github.com/YOUR_USERNAME/mtmc-tracking.git
git branch -M main
git push -u origin main
```

**Thay `YOUR_USERNAME`** bằng username GitHub của bạn.

### Bước 8: Verify

Mở GitHub repository và kiểm tra:
- ✅ Có 20+ files được push
- ✅ README.md hiển thị đẹp
- ✅ Không có files .pth/.mp4 (đã bị ignore)

## 📦 Xử lý Model Weights

Model weights **KHÔNG** được push lên GitHub (quá lớn + violate storage limits).

### Option 1: Hướng dẫn download trong README (Đã có)

README.md đã có section:
```markdown
### Download Pretrained Models

**OSNet Re-ID Model:**
- Download: [Google Drive](https://drive.google.com/...)
- Place in project root: `osnet_x1_0_market_256x128_*.pth`
```

### Option 2: Git LFS (Large File Storage) - Nếu cần

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model weights"

# Push with LFS
git push origin main
```

**Lưu ý:** GitHub LFS có giới hạn:
- Free: 1GB storage, 1GB bandwidth/month
- Paid: Cần upgrade plan

### Option 3: Host trên Google Drive/Dropbox (Khuyến nghị)

Đã setup trong README với link Google Drive.

## 🔄 Future Updates

### Khi có thay đổi mới:

```bash
# 1. Check status
git status

# 2. Add changes
git add .

# 3. Commit with message
git commit -m "Description of changes"

# 4. Push
git push origin main
```

### Ví dụ commits:

```bash
# Fix bug
git commit -m "Fix: Memory leak in Step 3 feature extraction"

# Add feature
git commit -m "Feat: Add real-time streaming support"

# Update docs
git commit -m "Docs: Add example results and visualizations"

# Refactor
git commit -m "Refactor: Optimize tracking speed by 20%"
```

## 👥 Team Collaboration

### Clone repository (team members):

```bash
git clone https://github.com/YOUR_USERNAME/mtmc-tracking.git
cd mtmc-tracking
```

### Pull latest changes:

```bash
git pull origin main
```

### Create feature branch:

```bash
# Create and switch to new branch
git checkout -b feature/your-feature-name

# Make changes...

# Commit
git add .
git commit -m "Add new feature"

# Push branch
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

## 🌟 Add Topics/Tags on GitHub

Sau khi push, vào GitHub repository → Settings → Topics:

Thêm tags:
```
computer-vision
object-tracking
multi-target-tracking
person-tracking
re-identification
deep-learning
pytorch
yolov8
botsort
mtmc
```

## 📝 Checklist sau khi push

- [ ] Repository hiển thị đúng trên GitHub
- [ ] README.md render đẹp với badges
- [ ] No model weights in repo (check file sizes)
- [ ] Team members có thể clone và setup theo SETUP_GUIDE.md
- [ ] Add topics/tags
- [ ] Update repository description
- [ ] (Optional) Add GitHub Actions for CI/CD
- [ ] (Optional) Enable GitHub Pages for documentation
- [ ] (Optional) Add CONTRIBUTING.md

## 🔗 Useful Links

- GitHub Desktop: https://desktop.github.com/ (GUI tool)
- GitHub CLI: https://cli.github.com/ (command line tool)
- Git documentation: https://git-scm.com/doc

## ⚠️ Common Issues

### Issue: "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/mtmc-tracking.git
```

### Issue: Push rejected (large files)

```bash
# Remove large files from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch *.pth *.pt" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (CAREFUL!)
git push origin main --force
```

### Issue: Authentication failed

GitHub không còn hỗ trợ password authentication.

**Solution: Use Personal Access Token**

1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `workflow`
4. Copy token
5. Use token as password khi push

**Or: Use SSH**

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub
cat ~/.ssh/id_ed25519.pub
# Copy and add to GitHub → Settings → SSH Keys

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/mtmc-tracking.git
```

---

**Chúc may mắn với GitHub repository! 🎉**

Nếu có vấn đề, check: https://docs.github.com/en/get-started
