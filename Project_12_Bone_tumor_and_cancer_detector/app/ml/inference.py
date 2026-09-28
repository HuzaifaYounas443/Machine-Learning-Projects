import os
import uuid

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .model_arch import ConvNeXtModel
from .gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple module-level cache so the (large) checkpoints are only loaded once
# per process instead of on every request.
_stage1_model = None
_stage2_model = None


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}\n"
            "Place your trained .pth file there or update the path in config.py / .env"
        )

    model = ConvNeXtModel(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_models(stage1_path, stage2_path=None):
    """Lazily load the requested models and cache them per process."""
    global _stage1_model, _stage2_model
    if _stage1_model is None:
        _stage1_model = load_model(stage1_path)
    if stage2_path and _stage2_model is None:
        _stage2_model = load_model(stage2_path)
    return _stage1_model, _stage2_model


def preprocess_image(image_path, apply_clahe=True, size=320):
    """
    Same preprocessing used during training/evaluation:
    CLAHE contrast enhancement -> resize -> ImageNet normalization.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    original_image = image.copy()

    transform = A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    return tensor, original_image


def predict_with_gradcam(image_path, model, class_names, target_layer, apply_clahe=True):
    input_tensor, original_image = preprocess_image(image_path, apply_clahe)

    gradcam = GradCAM(model, target_layer)
    try:
        cam, pred_class, output = gradcam.generate_cam(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        confidence = float(probs[pred_class])
        overlay, _heatmap = gradcam.overlay_heatmap(original_image, cam, alpha=0.45)
    finally:
        gradcam.remove_hooks()

    return {
        "prediction": class_names[pred_class],
        "prediction_class": int(pred_class),
        "confidence": confidence,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "overlay_bgr": overlay,
    }


def predict(image_path, model, class_names, apply_clahe=True):
    """Run a prediction without gradient tracking for the fast detection path."""
    input_tensor, _original_image = preprocess_image(image_path, apply_clahe)

    with torch.inference_mode():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    return {
        "prediction": class_names[pred_class],
        "prediction_class": pred_class,
        "confidence": float(probs[pred_class]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def save_overlay(overlay_bgr, out_path):
    cv2.imwrite(out_path, overlay_bgr)


def run_pipeline(
    image_path,
    stage1_model,
    stage2_model,
    results_folder,
    target_layer,
    stage2_path=None,
):
    """
    Full two-stage pipeline:
      Stage 1 - Tumor vs No Tumor
      Stage 2 - Benign vs Malignant (only runs if Stage 1 finds a tumor)
    Saves a Grad-CAM overlay image for each stage that runs and returns a
    dict ready to be stored on a Scan record.
    """
    run_id = uuid.uuid4().hex[:10]

    stage1_result = predict(image_path, stage1_model, ["No Tumor", "Tumor"])
    tumor_confidence = stage1_result["probabilities"]["Tumor"]
    tumor_detected = tumor_confidence > 0.70

    stage1_gradcam_name = None
    if tumor_detected:
        stage1_gradcam = predict_with_gradcam(
            image_path, stage1_model, ["No Tumor", "Tumor"], target_layer
        )
        stage1_gradcam_name = f"gradcam_stage1_{run_id}.jpg"
        save_overlay(
            stage1_gradcam["overlay_bgr"],
            os.path.join(results_folder, stage1_gradcam_name),
        )

    output = {
        "tumor_detected": tumor_detected,
        "tumor_confidence": tumor_confidence,
        "tumor_probabilities": stage1_result["probabilities"],
        "tumor_gradcam": stage1_gradcam_name,
        "classification": None,
        "classification_confidence": None,
        "classification_probabilities": None,
        "classification_gradcam": None,
    }

    if output["tumor_detected"]:
        if stage2_model is None:
            if not stage2_path:
                raise ValueError("Stage 2 model path is required for tumor classification")
            stage2_model = load_model(stage2_path)
            global _stage2_model
            _stage2_model = stage2_model

        stage2_result = predict_with_gradcam(
            image_path, stage2_model, ["Benign", "Malignant"], target_layer
        )
        stage2_gradcam_name = f"gradcam_stage2_{run_id}.jpg"
        save_overlay(
            stage2_result["overlay_bgr"], os.path.join(results_folder, stage2_gradcam_name)
        )

        output["classification"] = stage2_result["prediction"]
        output["classification_confidence"] = stage2_result["confidence"]
        output["classification_probabilities"] = stage2_result["probabilities"]
        output["classification_gradcam"] = stage2_gradcam_name

    return output
