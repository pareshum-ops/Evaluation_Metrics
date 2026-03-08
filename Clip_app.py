import os
import onnxruntime as ort

# Detect CUDA availability through ONNXRuntime BEFORE importing insightface
cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()

if cuda_available:
    print("GPU detected — enabling CUDA for InsightFace")
    os.environ["INSIGHTFACE_DISABLE_CUDA"] = "0"
else:
    print("No GPU detected — forcing InsightFace to CPU")
    os.environ["INSIGHTFACE_DISABLE_CUDA"] = "1"

import gradio as gr
import cv2
import numpy as np
import insightface
from numpy.linalg import norm

import torch
import clip
from PIL import Image

# -----------------------------
# Load ArcFace model once
# -----------------------------
_arcface_app = None

def load_arcface():
    global _arcface_app
    if _arcface_app is not None:
        return _arcface_app

    try:
        _arcface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _arcface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("ArcFace: GPU enabled")
    except Exception:
        print("ArcFace: GPU unavailable, using CPU")
        _arcface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        _arcface_app.prepare(ctx_id=-1, det_size=(640, 640))

    return _arcface_app

# -----------------------------
# Load CLIP model once
# -----------------------------
_clip_model = None
_clip_preprocess = None

def load_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return _clip_model, _clip_preprocess

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    print(f"CLIP loaded on {device}")
    return _clip_model, _clip_preprocess

# -----------------------------
# Utility: PIL → BGR
# -----------------------------
def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# -----------------------------
# Detect faces and sort left→right
# -----------------------------
def detect_and_sort_faces(img_bgr, expected_faces=2):
    app = load_arcface()
    faces = app.get(img_bgr)

    if len(faces) < expected_faces:
        return None

    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])
    return faces_sorted[:expected_faces]

# -----------------------------
# Extract embedding
# -----------------------------
def extract_embedding(face):
    return face.normed_embedding.astype(np.float32)

# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8))

# -----------------------------
# Main comparison function (Dual Identity)
# -----------------------------
def compare_faces(img1, img2, comparison_mode):
    img1_bgr = pil_to_bgr(img1)
    img2_bgr = pil_to_bgr(img2)

    faces1 = detect_and_sort_faces(img1_bgr, expected_faces=2)
    faces2 = detect_and_sort_faces(img2_bgr, expected_faces=2)

    if faces1 is None or faces2 is None:
        return "❌ Each image must contain TWO faces for dual-identity mode."

    left1, right1 = faces1
    left2, right2 = faces2

    emb_left1 = extract_embedding(left1)
    emb_right1 = extract_embedding(right1)
    emb_left2 = extract_embedding(left2)
    emb_right2 = extract_embedding(right2)

    if comparison_mode == "Left₁–Left₂ & Right₁–Right₂":
        return (
            f"Left₁ ↔ Left₂: {cosine_similarity(emb_left1, emb_left2):.4f}\n"
            f"Right₁ ↔ Right₂: {cosine_similarity(emb_right1, emb_right2):.4f}"
        )
    else:
        return (
            f"Left₁ ↔ Right₂: {cosine_similarity(emb_left1, emb_right2):.4f}\n"
            f"Right₁ ↔ Left₂: {cosine_similarity(emb_right1, emb_left2):.4f}"
        )

# -----------------------------
# Single Identity Comparison
# -----------------------------
def compare_single_identity(img1, img2):
    img1_bgr = pil_to_bgr(img1)
    img2_bgr = pil_to_bgr(img2)

    faces1 = detect_and_sort_faces(img1_bgr, expected_faces=1)
    faces2 = detect_and_sort_faces(img2_bgr, expected_faces=1)

    if faces1 is None or faces2 is None:
        return "❌ Each image must contain at least ONE face."

    emb1 = extract_embedding(faces1[0])
    emb2 = extract_embedding(faces2[0])

    score = cosine_similarity(emb1, emb2)
    return f"Single Identity Similarity: {score:.4f}"

# -----------------------------
# CLIP‑T: Text → Image similarity
# -----------------------------
def clip_text_similarity(prompt, img):
    if prompt is None or len(prompt.strip()) == 0:
        return None

    model, preprocess = load_clip()
    device = next(model.parameters()).device

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    img_p = preprocess(img).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(img_p)
        txt_emb = model.encode_text(text_tokens)

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    sim = (img_emb @ txt_emb.T).item()
    return float(sim)

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 ArcFace Identity Similarity + CLIP‑T Prompt Alignment")

    identity_mode = gr.Radio(
        ["Single Identity", "Dual Identity"],
        value="Single Identity",
        label="Identity Mode"
    )

    with gr.Row():
        img1 = gr.Image(label="Input Image 1")
        img2 = gr.Image(label="Input Image 2")

    prompt = gr.Textbox(
        label="Prompt (for CLIP‑T)",
        placeholder="Enter the text prompt used for generation"
    )

    comparison_mode = gr.Radio(
        ["Left₁–Left₂ & Right₁–Right₂", "Left₁–Right₂ & Right₁–Left₂"],
        value="Left₁–Left₂ & Right₁–Right₂",
        label="Comparison Mode",
        interactive=True,
        visible=False
    )

    output = gr.Textbox(label="Similarity Results", interactive=False)
    btn = gr.Button("Compute Similarity")

    def update_visibility(mode):
        if mode == "Single Identity":
            return gr.update(visible=False, interactive=False)
        else:
            return gr.update(visible=True, interactive=True)

    identity_mode.change(
        update_visibility,
        inputs=identity_mode,
        outputs=comparison_mode
    )

    def compute(img1_in, img2_in, mode, comp_mode, prompt_text):
        # ArcFace
        if mode == "Single Identity":
            arcface_result = compare_single_identity(img1_in, img2_in)
        else:
            arcface_result = compare_faces(img1_in, img2_in, comp_mode)

        # CLIP‑T
        clip_t_score = clip_text_similarity(prompt_text, img2_in) if prompt_text else None
        if clip_t_score is not None:
            clip_t_str = f"CLIP‑T (Prompt → Image₂): {clip_t_score:.4f}"
        else:
            clip_t_str = "CLIP‑T: No prompt provided."

        return f"{arcface_result}\n\n{clip_t_str}"

    btn.click(
        compute,
        inputs=[img1, img2, identity_mode, comparison_mode, prompt],
        outputs=output
    )

demo.launch(share=True)
