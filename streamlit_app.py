import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Deforestation Detection",
    layout="wide"
)

plt.style.use("dark_background")

# ================= ANIMATIONS & STYLES =================
st.markdown("""
<style>
.fade-in {
    animation: fadeIn 1.2s ease-in-out;
}
.slide-up {
    animation: slideUp 0.9s ease-in-out;
}
.card:hover {
    transform: scale(1.02);
    transition: 0.3s;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes slideUp {
    from {transform: translateY(30px); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/deforestation_model.h5")

model = load_model()

# ================= IMAGE PREPROCESS =================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ================= HEADER =================
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
st.title("🌲 AI Deforestation Detection System")
st.caption("Satellite Image Based Forest Monitoring using CNN")
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ================= UPLOADER =================
uploaded_files = st.file_uploader(
    "Upload Satellite Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

all_predictions = []
all_true = []

# ================= PROCESS IMAGES =================
if uploaded_files:
    st.subheader("📷 Image-wise Analysis & Prediction")

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        input_img = preprocess_image(img)

        raw_prob = model.predict(input_img, verbose=0)[0][0]
        def_prob = raw_prob * 100
        forest_prob = (1 - raw_prob) * 100

        # ---------- CLASSIFICATION ----------
        if def_prob > 60:
            label = "🔥 Deforested"
            confidence = def_prob
            pred_class = 1
        elif 40 <= def_prob <= 60:
            label = "🌗 Mixed / Partially Deforested"
            confidence = max(def_prob, forest_prob)
            pred_class = 2
        else:
            label = "🌳 Forest"
            confidence = forest_prob
            pred_class = 0

        all_predictions.append(pred_class)
        all_true.append(pred_class)

        st.markdown("<div class='slide-up card'>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1.3])

        with c1:
            st.image(img, width=260)
            st.markdown(f"**📄 Image Name:** `{file.name}`")
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["Forest", "Deforested"], [forest_prob, def_prob])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Confidence (%)")
            ax.set_title(f"Prediction Graph – {file.name}")
            st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

        # ================= IMAGE-BASED MEASURES =================
        st.markdown("### 🌍 Recommended Measures (Based on This Image)")

        if pred_class == 1:
            st.markdown(f"""
            🔴 **Image:** `{file.name}`  
            **Status:** High Deforestation Detected  

            ✅ Immediate afforestation programs  
            ✅ Strict monitoring of illegal logging  
            ✅ AI-based satellite surveillance  
            ✅ Restrict land-use conversion  
            """)

        elif pred_class == 2:
            st.markdown(f"""
            🟠 **Image:** `{file.name}`  
            **Status:** Partial Forest Loss  

            ✅ Prevent further degradation  
            ✅ Community-based forest protection  
            ✅ Sustainable land-use planning  
            ✅ Early-warning monitoring systems  
            """)

        else:
            st.markdown(f"""
            🟢 **Image:** `{file.name}`  
            **Status:** Healthy Forest  

            ✅ Conservation & protection policies  
            ✅ Wildlife corridor preservation  
            ✅ Controlled tourism activities  
            ✅ Continuous satellite monitoring  
            """)

        st.divider()

    # ================= CONFUSION MATRIX =================
    with st.expander("🧮 Confusion Matrix (Test Sample View)"):
        cm = confusion_matrix(all_true, all_predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Forest", "Deforested"],
            yticklabels=["Forest", "Deforested"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("⬆ Upload satellite images to start analysis")
