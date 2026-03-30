
import numpy as np
import cv2
import gradio as gr
from tensorflow.keras.models import load_model

# -------- LOAD MODELS (LOCAL FILES) --------
stage2 = load_model("deepfake_stage2_best.h5", compile=False)
stage3 = load_model("stage3_fm_model.h5", compile=False)
stage4 = load_model("stage4_fb_model.h5", compile=False)

def preprocess_image(img, size=64):
    img = img.astype("float32")
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    img = cv2.resize(img, (size, size))
    img = np.stack([img]*3, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

def predict(file):
    try:
        if file is None:
            return "❌ No file uploaded"

        path = file.name.lower()

        if path.endswith(".npy"):
            img = np.load(file.name, allow_pickle=True)
            if len(img.shape) == 3:
                img = img[:, :, 0]
        else:
            img = cv2.imread(file.name, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return "❌ Invalid image"

        img = preprocess_image(img)

        pred = float(stage2.predict(img, verbose=0)[0][0])

        if pred < 0.5:
            return f"✅ Real Image ({(1-pred)*100:.2f}%)"

        pred3 = float(stage3.predict(img, verbose=0)[0][0])

        if pred3 < 0.5:
            return f"⚠️ Fake (CTGAN - Malign) ({(1-pred3)*100:.2f}%)"
        else:
            return f"⚠️ Fake (Stable Diffusion - Malign) ({pred3*100:.2f}%)"

    except Exception as e:
        return f"❌ Error: {str(e)}"

interface = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload Image or .npy"),
    outputs="text",
    title="🧠 Deepfake Detector"
)

interface.launch()
