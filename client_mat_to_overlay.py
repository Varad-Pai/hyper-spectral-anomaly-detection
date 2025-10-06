import base64, io, argparse, os
from PIL import Image
import requests
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def load_mat_as_rgb_gt(mat_file, bands=(10,20,30)):
    mat = sio.loadmat(mat_file)
    if "data" not in mat or "map" not in mat:
        raise ValueError("MAT file must contain 'data' (HSI cube) and 'map' (GT mask)")

    data = mat["data"]  
    gt = mat["map"]      
    H, W, B = data.shape


    if max(bands) >= B:
        raise ValueError(f"Requested bands {bands}, but file has only {B} bands")
    rgb = data[:, :, list(bands)]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = (rgb * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb)

 
    gt_img = Image.fromarray((gt * 255).astype(np.uint8))

    return rgb_img, gt_img, gt



def call_api_mat(mat_path: str, url: str = "http://127.0.0.1:8000/predict_mat", bands=(10,20,30)):
    files = {
        "matfile": (os.path.basename(mat_path), open(mat_path, "rb"), "application/octet-stream")
    }
    data = {
        "bands": ",".join(str(x) for x in bands)
    }
    resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file", help="Path to .mat file (must contain 'data' and optionally 'map')")
    parser.add_argument("--bands", nargs=3, type=int, default=[10,20,30], help="Band indices for pseudo RGB")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict_mat", help="FastAPI endpoint URL")
    args = parser.parse_args()

  
    rgb_img, gt_img, gt_array = load_mat_as_rgb_gt(args.mat_file, tuple(args.bands))

   
    result = call_api_mat(args.mat_file, args.url, tuple(args.bands))

   
    overlay_b64 = result["overlay_png_base64"]
    overlay_img = Image.open(io.BytesIO(base64.b64decode(overlay_b64)))
    overlay_img.save("overlay_png_base64.png")
    print("✅ Overlay saved as overlay_png_base64.png")

    if "auc" in result:
        print("AUC from API:", result["auc"])

    
    try:
        mask_b64 = result.get("cleaned_mask_png_base64")
        if mask_b64 is not None and gt_array is not None:
            mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
            mask = (np.array(mask_img) > 127)
            gt_bin = (gt_array > 0)
            tp = int(np.logical_and(mask, gt_bin).sum())
            fp = int(np.logical_and(mask, ~gt_bin).sum())
            fn = int(np.logical_and(~mask, gt_bin).sum())
            iou = float(tp) / float(tp + fp + fn + 1e-6)
            
    except Exception as e:
        print("Warning: failed to compute metrics:", e)

    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(gt_array, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1,2,2)
    plt.imshow(overlay_img)
    plt.title("Overlay Result")
    plt.show()
