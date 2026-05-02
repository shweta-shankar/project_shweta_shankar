import torch
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y, mean, std, device
import os

def inferloader(list_of_img_paths):
    val_test_transforms = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    batch = []
    for path in list_of_img_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = val_test_transforms(img)
        batch.append(img_tensor)
    
    return torch.stack(batch).to(device)

def classify_images(model, list_of_img_paths, class_names):
    model.eval()
    model.to(device)
    
    # convert to input suitable for the model
    img_batch = inferloader(list_of_img_paths)
    
    with torch.no_grad():
        # predict the outcome
        logits = model(img_batch)
        preds = logits.argmax(dim=1)
        
    labels = [class_names[p] for p in preds]
    return labels
