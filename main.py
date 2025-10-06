from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import io, base64, os, tempfile
from typing import Optional, Dict, Any, Tuple, List

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import roc_curve, auc
from skimage import filters, morphology
from sklearn.covariance import LedoitWolf
import scipy.io as sio


class PatchCAE(nn.Module):
    def __init__(self, bands, latent_dim=64):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv2d(bands, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc1 = nn.Linear(64, latent_dim)
        self.fc2 = nn.Linear(latent_dim, bands)
    def forward(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.fc1(h)
        out = self.fc2(z)
        return out


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32)
    
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr

def numpy_to_pil(img: np.ndarray) -> Image.Image:
    
    if img.dtype != np.uint8:
        
        if img.max() <= 1.0:
            arr = (img * 255.0).clip(0,255).astype(np.uint8)
        else:
            arr = img.clip(0,255).astype(np.uint8)
    else:
        arr = img
    return Image.fromarray(arr)

def encode_png_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')

def extract_patches_and_centers(hsi: np.ndarray, patch_size: int=7, max_pixels: Optional[int]=None):
    
    H, W, B = hsi.shape
    pad = patch_size // 2
    padded = np.pad(hsi, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    patches = []
    centers = []

    for i in range(pad, pad+H):
        for j in range(pad, pad+W):
            patch = padded[i-pad:i+pad+1, j-pad:j+pad+1, :]  
            patch = np.transpose(patch, (2,0,1)).astype(np.float32)  
            center = padded[i,j,:].astype(np.float32)
            patches.append(patch)
            centers.append(center)
    patches = np.stack(patches, axis=0)
    centers = np.stack(centers, axis=0)
    if max_pixels is not None and patches.shape[0] > max_pixels:
        
        idx = np.random.choice(patches.shape[0], max_pixels, replace=False)
        patches = patches[idx]
        centers = centers[idx]
    return patches, centers


def spectral_angle_loss_torch(x, y):
    eps = 1e-8
    dot = torch.sum(x*y, dim=1)
    nx = torch.norm(x, dim=1)
    ny = torch.norm(y, dim=1)
    cos = dot / (nx*ny + eps)
    cos = torch.clamp(cos, -1+1e-7, 1-1e-7)
    angle = torch.acos(cos)
    return torch.mean(angle)


app = FastAPI(title="HSI Anomaly Detection (image input)")

MODEL_PATH = "patchcae_sad.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/predict")
async def predict(image: UploadFile = File(...), gt: Optional[UploadFile] = File(None),
                  patch_size: int = Form(7), max_dim: int = Form(256),
                  use_pretrained: bool = Form(True)) -> Dict[str, Any]:
    """
    Accepts:
      - image: image file (RGB or grayscale). We'll treat channels as 'bands'.
      - gt: optional ground truth binary mask image (same HxW)
      - patch_size: patch extraction size (odd, default 7)
      - max_dim: resize max dimension for speed (default 256)
      - use_pretrained: load model weights if available (default True)
    Returns JSON with base64-encoded PNGs: anomaly_map, cleaned_mask, overlay, (gt if provided), and auc if gt provided.
    """
    
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    
    w,h = img.size
    scale = 1.0
    if max(w,h) > max_dim:
        scale = max_dim / max(w,h)
        neww, newh = int(w*scale), int(h*scale)
        img = img.resize((neww,newh), Image.BILINEAR)

    arr = pil_to_numpy(img)  
    H, W, C = arr.shape

    
    gt_arr = None
    if gt is not None:
        gcont = await gt.read()
        gimg = Image.open(io.BytesIO(gcont)).convert("L")
        if scale != 1.0:
            gimg = gimg.resize((img.width, img.height), Image.NEAREST)
        gt_arr = np.array(gimg).astype(np.uint8)
        
        gt_bin = (gt_arr > 127).astype(np.uint8)
        gt_arr = gt_bin

    
    bands = arr.shape[2]
    data = arr / 255.0
    max_pixels = 20000  
    patches, centers = extract_patches_and_centers(data, patch_size=patch_size, max_pixels=max_pixels)
    N = patches.shape[0]

   
    band_means = centers.mean(axis=0, keepdims=True)
    band_stds  = centers.std(axis=0, keepdims=True) + 1e-6
    centers_norm = (centers - band_means) / band_stds
    patches_norm = (patches - band_means[:,:,None,None]) / band_stds[:,:,None,None]

    
    patches_t = torch.tensor(patches_norm, dtype=torch.float32, device=DEVICE)
    centers_t = torch.tensor(centers_norm, dtype=torch.float32, device=DEVICE)

    
    model = PatchCAE(bands=bands, latent_dim=64).to(DEVICE)

    
    if use_pretrained and os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
        except Exception as e:
            
            print("Failed loading pretrained weights:", e)

    
    if not any(p.numel() for p in model.parameters() if p.requires_grad) or (use_pretrained and not os.path.exists(MODEL_PATH)):
        
        pass

    
    if use_pretrained and not os.path.exists(MODEL_PATH):
       
        model.train()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        mse = nn.MSELoss()
        epochs = 8  
        batch_size = 1024
        for epoch in range(epochs):
            perm = np.random.permutation(N)
            running = 0.0
            steps = 0
            for start in range(0, N, batch_size):
                idx = perm[start:start+batch_size]
                xb = patches_t[idx]
                yb = centers_t[idx]
                noise = torch.randn_like(xb) * 0.01
                xb_noisy = xb + noise
                opt.zero_grad()
                pred = model(xb_noisy)
                loss_mse = mse(pred, yb)
                loss_sad = spectral_angle_loss_torch(pred, yb)
                loss = loss_mse + 1.0 * loss_sad
                loss.backward()
                opt.step()
                running += loss.item()
                steps += 1
            
        try:
            torch.save(model.state_dict(), MODEL_PATH)
        except Exception as e:
            print("Warning: failed to save quick weights:", e)
        model.eval()

    
    model.eval()
    with torch.no_grad():
        preds = []
        batch_size_inf = 1024
        
        total_pixels = H*W
        if total_pixels <= 50000:
            patches_full, centers_full = extract_patches_and_centers(data, patch_size=patch_size, max_pixels=None)
            band_means_full = centers_full.mean(axis=0, keepdims=True)
            band_stds_full = centers_full.std(axis=0, keepdims=True) + 1e-6
            patches_full_norm = (patches_full - band_means_full[:,:,None,None]) / band_stds_full[:,:,None,None]
            patches_full_t = torch.tensor(patches_full_norm, dtype=torch.float32, device=DEVICE)
            preds_list = []
            for start in range(0, patches_full_t.shape[0], batch_size_inf):
                xb = patches_full_t[start:start+batch_size_inf]
                p = model(xb)
                preds_list.append(p.cpu().numpy())
            preds_all = np.vstack(preds_list)
           
            preds_denorm = preds_all * band_stds_full + band_means_full
            centers_orig = centers_full
            
            mse_scores = np.mean((preds_denorm - centers_orig)**2, axis=1)
            dot = np.sum(preds_denorm * centers_orig, axis=1)
            n1 = np.linalg.norm(preds_denorm, axis=1)
            n2 = np.linalg.norm(centers_orig, axis=1)
            cos = np.clip(dot / (n1*n2 + 1e-12), -1+1e-12, 1-1e-12)
            angles = np.arccos(cos)
            angles_deg = angles * 180.0 / np.pi
            mse_z = stats.zscore(mse_scores)
            ang_z = stats.zscore(angles_deg)
            combined = mse_z + ang_z
            anom_map = combined.reshape(H, W)
        else:
            
            preds_list = []
            for start in range(0, patches_t.shape[0], batch_size_inf):
                xb = patches_t[start:start+batch_size_inf]
                p = model(xb)
                preds_list.append(p.cpu().numpy())
            preds_sample = np.vstack(preds_list)
            preds_denorm = preds_sample * band_stds + band_means
            centers_orig = centers
            mse_scores = np.mean((preds_denorm - centers_orig)**2, axis=1)
            dot = np.sum(preds_denorm * centers_orig, axis=1)
            n1 = np.linalg.norm(preds_denorm, axis=1)
            n2 = np.linalg.norm(centers_orig, axis=1)
            cos = np.clip(dot / (n1*n2 + 1e-12), -1+1e-12, 1-1e-12)
            angles = np.arccos(cos)
            angles_deg = angles * 180.0 / np.pi
            mse_z = stats.zscore(mse_scores)
            ang_z = stats.zscore(angles_deg)
            combined_sample = mse_z + ang_z
            
        
            full_flat = np.zeros(H*W, dtype=np.float32)
            
            idx = np.random.choice(H*W, size=combined_sample.shape[0], replace=False)
            full_flat[idx] = combined_sample
            anom_map = full_flat.reshape(H,W)

    
    score_norm = (anom_map - anom_map.min()) / (anom_map.max() - anom_map.min() + 1e-12)

    
    thresh = filters.threshold_otsu(score_norm)
    binary_map = score_norm > thresh
    binary_clean = morphology.remove_small_objects(binary_map, min_size=20)
    binary_clean = morphology.binary_opening(binary_clean, morphology.disk(2))

    
    def heatmap_to_rgb(img_norm):
        
        img = (img_norm * 255.0).astype(np.uint8)
        
        rgb = np.stack([img, (img*0.7).clip(0,255).astype(np.uint8), (255-img)], axis=-1)
        return rgb

    anom_rgb = heatmap_to_rgb(score_norm)
    anom_pil = numpy_to_pil(anom_rgb)

    clean_img = (binary_clean.astype(np.uint8) * 255)
    clean_pil = numpy_to_pil(clean_img)

    overlay = img.copy().convert("RGBA")
    overlay_arr = np.array(overlay).astype(np.uint8)
    mask_rgba = np.zeros((H,W,4), dtype=np.uint8)
    mask_rgba[...,0] = 255
    mask_rgba[...,3] = (binary_clean.astype(np.uint8) * 120)  
    overlay_img = Image.alpha_composite(overlay.convert("RGBA"), Image.fromarray(mask_rgba))
    overlay_pil = overlay_img.convert("RGB")

    anom_b64 = encode_png_base64(anom_pil)
    clean_b64 = encode_png_base64(clean_pil)
    overlay_b64 = encode_png_base64(overlay_pil)
    gt_b64 = None
    auc_value = None
    if gt_arr is not None:
        gt_pil = numpy_to_pil((gt_arr*255).astype(np.uint8))
        gt_b64 = encode_png_base64(gt_pil)
        try:
            y_true = gt_arr.flatten().astype(int)
            y_score = score_norm.flatten()
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = float(auc(fpr,tpr))
        except Exception as e:
            auc_value = None

    resp = {
        "anomaly_map_png_base64": anom_b64,
        "cleaned_mask_png_base64": clean_b64,
        "overlay_png_base64": overlay_b64,
        "height": H,
        "width": W,
        "bands_used": bands
    }
    if gt_b64 is not None:
        resp["gt_png_base64"] = gt_b64
    if auc_value is not None:
        resp["auc"] = auc_value

    return JSONResponse(content=resp)


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)

def compute_pseudo_rgb_from_cube(cube: np.ndarray, band_indices: Tuple[int,int,int]) -> Image.Image:
    r, g, b = band_indices
    H, W, B = cube.shape
    if max(band_indices) >= B:
        raise ValueError(f"Requested bands {band_indices}, but file has only {B} bands")
    rgb = cube[:, :, [r, g, b]].astype(np.float32)
    rgb = minmax_normalize(rgb)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_uint8)

def rx_scores_global(cube: np.ndarray, shrinkage: Optional[float]=None) -> np.ndarray:
    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float64)
    mean_vec = X.mean(axis=0)
    Xc = X - mean_vec
    if shrinkage is None:
        lw = LedoitWolf(store_precision=False, assume_centered=True)
        lw.fit(Xc)
        cov = lw.covariance_
    else:
        emp_cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        diag = np.diag(np.diag(emp_cov))
        cov = (1.0 - shrinkage) * emp_cov + shrinkage * diag
    try:
        cov_inv = np.linalg.pinv(cov)
    except Exception:
        cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(B))
    quad = np.einsum('ij,jk,ik->i', Xc, cov_inv, Xc)
    scores = quad.reshape(H, W).astype(np.float32)
    return scores

@app.post("/predict_mat")
async def predict_mat(matfile: UploadFile = File(...),
                      bands: str = Form("10,20,30"),
                      use_ledoitwolf: bool = Form(True),
                      shrinkage: float = Form(0.0),
                      post_min_size: int = Form(20)) -> Dict[str, Any]:
    """
    Accepts a .mat file containing 'data' (HSI cube, shape HxWxB) and optional 'map' (HxW binary GT).
    Runs robust RX anomaly detection with Ledoit-Wolf covariance.
    Returns base64-encoded PNGs: anomaly_map, cleaned_mask, overlay, optional gt, and AUC if GT exists.
    """
    content = await matfile.read()
    with io.BytesIO(content) as buf:
        mat = sio.loadmat(buf)
    if 'data' not in mat:
        raise ValueError("MAT must contain key 'data' with shape (H,W,B)")
    cube = mat['data']
    if cube.ndim != 3:
        raise ValueError("'data' must be 3D (H,W,B)")
    cube = cube.astype(np.float32)
    H, W, B = cube.shape

    gt_arr = None
    for key in ['map', 'gt', 'ground_truth']:
        if key in mat:
            gt_val = mat[key]
            if gt_val.ndim == 2 and gt_val.shape[:2] == (H, W):
                gt_arr = (gt_val > 0).astype(np.uint8)
                break

    shrink = None if use_ledoitwolf or shrinkage <= 0.0 else float(shrinkage)
    scores = rx_scores_global(cube, shrinkage=shrink)
    score_norm = minmax_normalize(scores)

    thresh = filters.threshold_otsu(score_norm)
    binary_map = score_norm > thresh
    binary_clean = morphology.remove_small_objects(binary_map, min_size=int(post_min_size))
    binary_clean = morphology.binary_opening(binary_clean, morphology.disk(2))

    try:
        band_tuple = tuple(int(x.strip()) for x in bands.split(','))
        if len(band_tuple) != 3:
            raise ValueError
    except Exception:
        band_tuple = (10, 20, 30)
    rgb_pil = compute_pseudo_rgb_from_cube(cube, band_tuple)

    overlay = rgb_pil.convert("RGBA")
    mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    mask_rgba[..., 0] = 255
    mask_rgba[..., 3] = (binary_clean.astype(np.uint8) * 120)
    overlay_img = Image.alpha_composite(overlay, Image.fromarray(mask_rgba))
    overlay_pil = overlay_img.convert("RGB")

    img8 = (score_norm * 255.0).astype(np.uint8)
    heat_rgb = np.stack([img8, (img8*0.7).clip(0,255).astype(np.uint8), (255-img8)], axis=-1)
    heat_pil = Image.fromarray(heat_rgb)

    
    anom_b64 = encode_png_base64(heat_pil)
    clean_b64 = encode_png_base64(Image.fromarray((binary_clean.astype(np.uint8)*255)))
    overlay_b64 = encode_png_base64(overlay_pil)

    gt_b64 = None
    auc_value = None
    if gt_arr is not None:
        gt_b64 = encode_png_base64(Image.fromarray((gt_arr*255).astype(np.uint8)))
        try:
            y_true = gt_arr.flatten().astype(int)
            y_score = score_norm.flatten()
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = float(auc(fpr, tpr))
        except Exception:
            auc_value = None

    resp = {
        "anomaly_map_png_base64": anom_b64,
        "cleaned_mask_png_base64": clean_b64,
        "overlay_png_base64": overlay_b64,
        "height": H,
        "width": W,
        "bands_used": 3,
        "pseudo_rgb_bands": list(band_tuple),
    }
    if gt_b64 is not None:
        resp["gt_png_base64"] = gt_b64
    if auc_value is not None:
        resp["auc"] = auc_value

    return JSONResponse(content=resp)
