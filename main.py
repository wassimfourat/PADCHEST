import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Chest X-ray Classification",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1200px; padding-top: 1rem;}
      header, footer {visibility: hidden;}
      .stProgress > div > div > div > div { transition: width 0.6s ease; }
      .caption-muted { color: #6b7280; }
      [data-testid="stImage"] img {
        max-height: 520px;
        width: auto !important;
        object-fit: contain;
      }
    </style>
    """,
    unsafe_allow_html=True
)

labels = ['normal', 'chronic changes', 'aortic elongation', 'scoliosis', 'COPD signs', 'cardiomegaly']

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal = alpha * focal_weight * bce
        return tf.reduce_mean(focal)
    return loss

@st.cache_resource
def load_model(model_path):
    metrics = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.AUC(name='auc', multi_label=True),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'loss': focal_loss(gamma=2.0, alpha=0.25),
            'accuracy': keras.metrics.BinaryAccuracy(name='accuracy'),
            'auc': keras.metrics.AUC(name='auc', multi_label=True),
            'precision': keras.metrics.Precision(name='precision'),
            'recall': keras.metrics.Recall(name='recall')
        }
    )
    return model

def preprocess_image(image, target_size=(224, 224)):
    img = np.array(image)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
    img_resized = cv2.resize(img_rgb, target_size)
    img_display = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_display

def predict_image(model, img_array, threshold=0.5):
    preds = model.predict(img_array, verbose=0)[0]
    predictions = []
    for i, prob in enumerate(preds):
        if prob >= threshold:
            predictions.append((labels[i], prob))
    return preds, predictions

def main():
    st.title("Chest X-ray Classification")
    st.markdown(
        "<p class='caption-muted'>Upload a chest radiograph to obtain model-predicted probabilities for predefined findings.</p>",
        unsafe_allow_html=True
    )

    st.sidebar.header("Configuration")
    model_path = "best_padchest_model.h5"
    threshold = st.sidebar.slider(
        "Prediction threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Minimum probability required for a finding to be listed under Predictions."
    )

    try:
        model = load_model(model_path)
        st.sidebar.info("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        tab_img, tab_results = st.tabs(["Image", "Results"])

        with st.spinner("Analyzing image"):
            img_array, img_display = preprocess_image(image)
            preds, predictions = predict_image(model, img_array, threshold)

        with tab_img:
            st.subheader("Input image")
            st.image(
                img_display,
                channels="RGB",
                caption="Displayed with normalized contrast for viewing only.",
                use_column_width=False,
                width=720
            )

        with tab_results:
            st.subheader("Analysis results")
            st.markdown("**Class probabilities**")
            for i, prob in enumerate(preds):
                st.progress(float(prob), text=f"{labels[i]} — {prob:.4f} ({prob*100:.2f}%)")
            st.markdown("---")
            st.markdown(f"**Predictions (threshold ≥ {threshold:.2f})**")
            if predictions:
                for label, prob in predictions:
                    st.write(f"- **{label}**: {prob*100:.2f}%")
            else:
                st.write("No findings met the specified threshold.")
            with st.expander("Detailed probability table"):
                import pandas as pd
                df = pd.DataFrame({
                    'Condition': labels,
                    'Probability': preds,
                    'Percentage': [f"{p*100:.2f}%" for p in preds],
                    'Above Threshold': ['Yes' if p >= threshold else 'No' for p in preds]
                }).sort_values('Probability', ascending=False)
                st.dataframe(df, use_container_width=True)
                st.caption("This tool reports model outputs and does not replace clinical judgment.")
    else:
        st.info("Please upload a chest X-ray to begin.")
        with st.expander("Instructions"):
            st.markdown("""
            1. Upload a chest radiograph (PNG/JPG).
            2. Adjust the prediction threshold in the sidebar if needed.
            3. Review probabilities for each class and thresholded predictions.

            Classes evaluated:
            - Normal
            - Chronic changes
            - Aortic elongation
            - Scoliosis
            - COPD signs
            - Cardiomegaly
            """)

if __name__ == "__main__":
    main()
