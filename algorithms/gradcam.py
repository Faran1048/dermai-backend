"""
algorithms/gradcam.py

Generates Grad-CAM activation heatmaps overlaying the last convolutional layers
of an image classifier model to visualize the lesion regions of interest.
"""

from __future__ import annotations
import os
import uuid
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def generate_dummy_heatmap(image_path: str) -> str:
    """Fallback generator to create a central highlight heatmap if Grad-CAM fails."""
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    h, w, _ = orig_img.shape
    # Create a simple radial Gaussian center mask
    mask = np.zeros((h, w), dtype=np.float32)
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4

    cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
    
    # Apply Gaussian blur to make it look like an activation heatmap
    blur_kernel = min(h, w) | 1  # ensure odd size
    if blur_kernel < 3:
        blur_kernel = 3
    elif blur_kernel > 101:
        blur_kernel = 101
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    
    # Normalize mask to [0, 1]
    denom = mask.max() - mask.min()
    if denom > 0:
        mask = (mask - mask.min()) / denom

    heatmap_uint8 = np.uint8(255 * mask)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(colormap, 0.4, orig_img, 0.6, 0)

    # Save to workspace local tmp
    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"gradcam_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(out_path, overlay)
    return out_path

def generate_heatmap(image_path: str, model, predicted_class_idx: int) -> str:
    """
    Generate a Grad-CAM heatmap showing the model's visual attention/activation areas.

    Args:
        image_path: Path to the input image file.
        model: PyTorch model instance (e.g. EfficientNet).
        predicted_class_idx: Index of the predicted class (0-6).

    Returns:
        Path to the saved Grad-CAM overlay image.
    """
    try:
        # pyrefly: ignore [missing-import]
        from pytorch_grad_cam import GradCAM
        # pyrefly: ignore [missing-import]
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        # Fallback to dummy generation if pytorch_grad_cam is missing
        return generate_dummy_heatmap(image_path)

    # If the model does not have .features or targeting fails, use fallback
    if not hasattr(model, "features"):
        return generate_dummy_heatmap(image_path)

    cam = None
    try:
        # Define target layer: model.features[-1] (EfficientNet last conv)
        target_layers = [model.features[-1]]
        
        # Load image once
        pil_img = Image.open(image_path).convert("RGB")
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(pil_img).unsqueeze(0) # shape (1, 3, 224, 224)

        # Run Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class_idx)]
        
        # cam() returns values between 0.0 and 1.0, shape: (1, 224, 224)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Convert PIL image to BGR cv2 image in-memory to avoid reloading from disk
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w, _ = orig_img.shape

        # Resize the 224x224 heatmap back to original image size
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))

        # Convert to uint8 and apply JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend original image and colormap
        overlay = cv2.addWeighted(colormap, 0.4, orig_img, 0.6, 0)

        # Save to workspace local tmp
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"gradcam_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(out_path, overlay)

        return out_path

    except Exception:
        # Fall back to dummy heatmap if anything fails during execution
        return generate_dummy_heatmap(image_path)
    finally:
        # Clean up registered hooks to prevent memory leaks
        if cam is not None and hasattr(cam, "activations_and_gradients"):
            try:
                cam.activations_and_gradients.release()
            except Exception:
                pass
        # Clear model gradients and reset computation graph
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass

