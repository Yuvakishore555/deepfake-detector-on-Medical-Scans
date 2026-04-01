import numpy as np
import cv2
import pickle
import gradio as gr
from tensorflow.keras.models import load_model

# =========================
# LOAD MODELS
# =========================
print("🔄 Loading models...")

stage2_model = load_model("stage2_real_fake_320.keras")
stage3_model = load_model("stage3_ctgan_sd.keras")
stage4_model = load_model("stage4_injection_removal.keras")
stage5_model = load_model("stage5_tm_tb.keras")

with open("csv_model.pkl", "rb") as f:
    csv_model = pickle.load(f)

print("✅ All models loaded successfully")


# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(path):
    img = np.load(path)

    img = img.astype("float32")
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    img = cv2.resize(img, (320, 320))
    img = np.stack([img]*3, axis=-1)

    return np.expand_dims(img, axis=0)


# =========================
# CSV PREDICTION
# =========================
def predict_csv(features):
    pred = csv_model.predict([features])[0]
    return "Fake" if pred == 1 else "Real"


# =========================
# IMAGE PIPELINE
# =========================
def predict_image(path):
    img = preprocess_image(path)

    # Stage 2
    s2 = stage2_model.predict(img)[0][0]
    real_fake = "Fake" if s2 > 0.5 else "Real"

    result = {
        "Stage2_RealFake": real_fake
    }

    if real_fake == "Real":
        return result

    # Stage 3
    s3 = stage3_model.predict(img)[0][0]
    gan_type = "CTGAN" if s3 > 0.5 else "Stable Diffusion"

    # Stage 4
    s4 = stage4_model.predict(img)[0][0]
    manipulation = "Injection" if s4 > 0.5 else "Removal"

    # Stage 5
    s5 = stage5_model.predict(img)[0][0]
    tumor = "Malignant" if s5 > 0.5 else "Benign"

    result.update({
        "Stage3_GAN": gan_type,
        "Stage4_Manipulation": manipulation,
        "Stage5_Tumor": tumor
    })

    return result


# =========================
# MAIN FUNCTION
# =========================
def predict(input_data, mode="image"):
    if mode == "csv":
        return predict_csv(input_data)
    else:
        return predict_image(input_data)


# =========================
# GRADIO FUNCTIONS
# =========================
def gradio_image_predict(file):
    if file is None:
        return "Please upload a .npy file"

    try:
        result = predict(file.name, mode="image")
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def gradio_csv_predict(x, y, scanner, cur_slice):
    try:
        features = [x, y, scanner, cur_slice]
        result = predict(features, mode="csv")
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# GRADIO UI
# =========================
with gr.Blocks() as app:
    gr.Markdown("# 🧠 Medical AI Pipeline")
    gr.Markdown("Multi-stage CT Scan Analysis System")

    with gr.Tab("📁 Image Prediction (.npy)"):
        file_input = gr.File(label="Upload .npy file")
        img_output = gr.Textbox(label="Result")
        btn1 = gr.Button("Predict")

        btn1.click(gradio_image_predict,
                   inputs=file_input,
                   outputs=img_output)

    with gr.Tab("📊 CSV Prediction"):
        x = gr.Number(label="x")
        y = gr.Number(label="y")
        scanner = gr.Number(label="scanner")
        cur_slice = gr.Number(label="cur_slice")

        csv_output = gr.Textbox(label="Result")
        btn2 = gr.Button("Predict")

        btn2.click(gradio_csv_predict,
                   inputs=[x, y, scanner, cur_slice],
                   outputs=csv_output)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.launch()
