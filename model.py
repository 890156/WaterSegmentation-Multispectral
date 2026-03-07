import torch
import segmentation_models_pytorch as smp
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    
    if not os.path.exists("water_seg_model.pth"):
        raise FileNotFoundError("Model file not found!")  

    
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=12,
        classes=1
    )

    model.load_state_dict(
        torch.load("water_seg_model.pth", map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model