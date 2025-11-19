@echo off
REM Batch processing for multi-camera MTMC pipeline (Windows)

REM Configuration
set DEVICE=0
set MAX_CROPS=10
set SIMILARITY_THRESHOLD=0.6

REM Camera videos (THAY ĐỔI PATHS NÀY)
set CAM1_NAME=camera1
set CAM1_PATH=D:\Videos\camera1.mp4

set CAM2_NAME=camera2
set CAM2_PATH=D:\Videos\camera2.mp4

set CAM3_NAME=camera3
set CAM3_PATH=D:\Videos\camera3.mp4

echo =========================================
echo MTMC PIPELINE - BATCH PROCESSING
echo =========================================
echo.

REM ==========================================
REM CAMERA 1
REM ==========================================
echo =========================================
echo Processing: %CAM1_NAME%
echo Video: %CAM1_PATH%
echo =========================================

echo [Step 2] Running tracking...
python step2_tracking.py --source "%CAM1_PATH%" --output-dir runs/%CAM1_NAME%_tracking --device %DEVICE%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Tracking failed for %CAM1_NAME%
    exit /b 1
)

REM Find tracks file
for /r "runs\%CAM1_NAME%_tracking" %%f in (*.txt) do (
    set CAM1_TRACKS=%%f
    goto :found_cam1_tracks
)
:found_cam1_tracks
echo [INFO] Tracks: %CAM1_TRACKS%

echo [Step 3] Extracting Re-ID features...
python step3_reid_extraction.py --source "%CAM1_PATH%" --tracks "%CAM1_TRACKS%" --output-dir runs/%CAM1_NAME%_features --device %DEVICE% --max-crops-per-track %MAX_CROPS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Feature extraction failed for %CAM1_NAME%
    exit /b 1
)

echo [INFO] √ %CAM1_NAME% completed
echo.

REM ==========================================
REM CAMERA 2
REM ==========================================
echo =========================================
echo Processing: %CAM2_NAME%
echo Video: %CAM2_PATH%
echo =========================================

echo [Step 2] Running tracking...
python step2_tracking.py --source "%CAM2_PATH%" --output-dir runs/%CAM2_NAME%_tracking --device %DEVICE%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Tracking failed for %CAM2_NAME%
    exit /b 1
)

REM Find tracks file
for /r "runs\%CAM2_NAME%_tracking" %%f in (*.txt) do (
    set CAM2_TRACKS=%%f
    goto :found_cam2_tracks
)
:found_cam2_tracks
echo [INFO] Tracks: %CAM2_TRACKS%

echo [Step 3] Extracting Re-ID features...
python step3_reid_extraction.py --source "%CAM2_PATH%" --tracks "%CAM2_TRACKS%" --output-dir runs/%CAM2_NAME%_features --device %DEVICE% --max-crops-per-track %MAX_CROPS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Feature extraction failed for %CAM2_NAME%
    exit /b 1
)

echo [INFO] √ %CAM2_NAME% completed
echo.

REM ==========================================
REM CAMERA 3
REM ==========================================
echo =========================================
echo Processing: %CAM3_NAME%
echo Video: %CAM3_PATH%
echo =========================================

echo [Step 2] Running tracking...
python step2_tracking.py --source "%CAM3_PATH%" --output-dir runs/%CAM3_NAME%_tracking --device %DEVICE%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Tracking failed for %CAM3_NAME%
    exit /b 1
)

REM Find tracks file
for /r "runs\%CAM3_NAME%_tracking" %%f in (*.txt) do (
    set CAM3_TRACKS=%%f
    goto :found_cam3_tracks
)
:found_cam3_tracks
echo [INFO] Tracks: %CAM3_TRACKS%

echo [Step 3] Extracting Re-ID features...
python step3_reid_extraction.py --source "%CAM3_PATH%" --tracks "%CAM3_TRACKS%" --output-dir runs/%CAM3_NAME%_features --device %DEVICE% --max-crops-per-track %MAX_CROPS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Feature extraction failed for %CAM3_NAME%
    exit /b 1
)

echo [INFO] √ %CAM3_NAME% completed
echo.

REM ==========================================
REM STEP 4: INTER-CAMERA ASSOCIATION
REM ==========================================
echo =========================================
echo [Step 4] Running inter-camera association
echo =========================================

python step4_inter_camera_association.py --features runs/%CAM1_NAME%_features/track_features.pkl runs/%CAM2_NAME%_features/track_features.pkl runs/%CAM3_NAME%_features/track_features.pkl --camera-names %CAM1_NAME% %CAM2_NAME% %CAM3_NAME% --method hungarian --similarity-threshold %SIMILARITY_THRESHOLD% --output-dir runs/mtmc_results --visualize

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Inter-camera association failed
    exit /b 1
)

echo.
echo =========================================
echo PIPELINE COMPLETED!
echo =========================================
echo Results saved to: runs\mtmc_results\
echo.
echo Output files:
echo   - global_id_mapping.json
echo   - pairwise_matches.json
echo   - similarity_*.png (heatmaps)
echo.
echo To view global ID mapping:
echo   type runs\mtmc_results\global_id_mapping.json
echo.

pause
