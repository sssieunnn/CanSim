import torch
from torchvision import transforms
from torch import amp
from PIL import ImageDraw
import numpy as np
import os

def available_drugs_for(cancer_type: int, drug_df):
    sub = drug_df[drug_df["cancer_type"] == cancer_type]
    return sorted(sub["drug_type"].unique().tolist())[:4]

def simulate_all_drugs_fast(model, base_image, tabular_tensor, device, cancer_type, drug_map, drug_df, num_weeks=12):
    tform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    valid_drugs = available_drugs_for(int(cancer_type), drug_df)

    week_images = []
    for w in range(num_weeks):
        img_np = np.array(base_image).astype(np.float32)
        tumor_radius = 40 - int((w / num_weeks) * 30)
        yy, xx = np.ogrid[:img_np.shape[0], :img_np.shape[1]]
        cy, cx = img_np.shape[0] // 2, img_np.shape[1] // 2
        dist = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = np.ones_like(img_np); mask[dist < tumor_radius**2] = 0.7
        img_np = np.clip(img_np * mask, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img)
        draw.ellipse([(cx - tumor_radius, cy - tumor_radius),
                      (cx + tumor_radius, cy + tumor_radius)], outline="red", width=2)
        week_images.append(img)

    week_probs = []
    model.eval()
    with torch.no_grad():
        for wimg in week_images:
            x = tform(wimg).unsqueeze(0).to(device)
            tab = tabular_tensor.unsqueeze(0).to(device)
            with amp.autocast('cuda' if device.type=='cuda' else 'cpu', dtype=torch.float16):
                logits = model(x, tab).squeeze(0)
                probs = torch.softmax(logits, dim=0).cpu().numpy()
            week_probs.append(probs)

    results = []
    for d in valid_drugs:
        dname = drug_map.get((int(cancer_type), int(d)), f"Drug_{d}")
        scores = [float(p[d]) for p in week_probs]
        results.append({
            "drug": dname,
            "drug_idx": int(d),
            "images": week_images,
            "scores": scores,
            "final_score": scores[-1]
        })
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    best_drug = max(results, key=lambda x: x["final_score"])
    return results, best_drug