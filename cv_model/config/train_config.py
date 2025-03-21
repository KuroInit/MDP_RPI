
train_config = {
    # --------------------------------
    # Basic
    # --------------------------------
    # "data": "/p/data1/mmlaion/y4/MDP_CV-2/data.yaml",   
    # "data": "/p/data1/mmlaion/y4/MDP_CV-2/data_with_bg.yaml",   
    "data": "/p/data1/mmlaion/y4/MDP_CV-2/data_edge.yaml",
    "project": "MDP",  
    "name": "yolo11n_v5_fused_aug_bs128",               
    
    # --------------------------------
    # Training
    # --------------------------------
    "model": "yolo11n.pt",    # Pretrained model
    "resume": None,          # Resume training if needed
    "epochs": 100,            
    "batch_size": 128,       
    "img_size": 640,          
    "optimizer": "auto",      
    "lr0": 1e-3,              
    "seed": 42,             

    # --------------------------------
    # Multi gpu
    # --------------------------------
    "device": "0,1,2,3",     

    # --------------------------------
    # Other
    # --------------------------------
    "workers": 1,            
    "verbose": True,          
    "save_period": 10,  
}  