import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

perceptron = tf.keras.models.load_model("perceptron_model_digit.keras")
ann = tf.keras.models.load_model("ann_model_digit.keras")
cnn = tf.keras.models.load_model("cnn_model_digit.keras")

st.set_page_config(page_title="Digit Recognition", page_icon="🔢", layout="centered")

st.title("Handwritten Digit Recognition")
st.write("Draw a digit below and compare predictions from Perceptron, ANN and CNN models.")

def preprocess_digit(image):
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    img = np.array(image)

    if np.mean(img) > 127:
        img = 255 - img

    img = Image.fromarray(img.astype(np.uint8))

    bbox = img.getbbox()

    if bbox:
        img = img.crop(bbox)

    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (28, 28), 0)

    left = (28 - img.size[0]) // 2
    top = (28 - img.size[1]) // 2

    new_img.paste(img, (left, top))

    arr = np.array(new_img).astype("float32") / 255.0

    return arr, new_img

st.subheader("Draw Digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict Digit"):

    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)

        img = Image.fromarray(img)
        img = img.convert("RGB")

        processed_array, processed_image = preprocess_digit(img)

        X_img = processed_array.reshape(1, 28, 28)
        X_cnn = processed_array.reshape(1, 28, 28, 1)

        pred_percp = perceptron.predict(X_img, verbose=0)
        pred_ann = ann.predict(X_img, verbose=0)
        pred_cnn = cnn.predict(X_cnn, verbose=0)

        percp_digit = np.argmax(pred_percp)
        ann_digit = np.argmax(pred_ann)
        cnn_digit = np.argmax(pred_cnn)

        percp_conf = np.max(pred_percp) * 100
        ann_conf = np.max(pred_ann) * 100
        cnn_conf = np.max(pred_cnn) * 100

        st.subheader("Model Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Perceptron", int(percp_digit))
            st.write(f"Confidence: {percp_conf:.2f}%")

        with col2:
            st.metric("ANN", int(ann_digit))
            st.write(f"Confidence: {ann_conf:.2f}%")

        with col3:
            st.metric("CNN", int(cnn_digit))
            st.write(f"Confidence: {cnn_conf:.2f}%")

        final_digit = int(cnn_digit)

        st.subheader("Final Result")
        st.success(f"Final Predicted Digit: {final_digit}")

    else:
        st.warning("Please draw a digit first.")