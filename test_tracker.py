from boxmot import track

track(
    model='yolov8n.pt',                   
    source='people-walking.mp4',            
    tracker_type='botsort',                 
    
    with_reid=True,                        
    reid_model='osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',  
    save=True,
    show=True
)

print("Đã chạy xong.")