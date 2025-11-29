"""
Skin Lesion Classification App
8-Class Dermoscopic Image Classifier using ISIC2019-trained EfficientNet-B4
"""

# ✅ STREAMLIT MUST BE IMPORTED FIRST
import streamlit as st

# Then all other imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import requests
from io import BytesIO
import numpy as np
import plotly.graph_objects as go
import cv2
import base64  # Added for heatmap animation

# -------------------------
# Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/Skindoc/streamlit5/resolve/main/best_model_20251116_151842.pth"
MODEL_NAME = "tf_efficientnet_b4"
NUM_CLASSES = 8
IMG_SIZE = 384

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

CLASS_INFO = {
    'akiec': {'full_name': 'Actinic Keratoses (AKIEC)', 'description': 'Pre-cancerous lesions caused by sun damage. Requires monitoring and treatment.', 'risk': 'Medium', 'color': '#FFA500'},
    'bcc': {'full_name': 'Basal Cell Carcinoma (BCC)', 'description': 'Most common skin cancer. Slow-growing, rarely spreads, highly treatable.', 'risk': 'High', 'color': '#FF4444'},
    'bkl': {'full_name': 'Benign Keratosis (BKL)', 'description': 'Non-cancerous skin growth. Generally harmless but may be removed for cosmetic reasons.', 'risk': 'Low', 'color': '#90EE90'},
    'df': {'full_name': 'Dermatofibroma (DF)', 'description': 'Benign fibrous nodule. Usually harmless and does not require treatment.', 'risk': 'Low', 'color': '#87CEEB'},
    'mel': {'full_name': 'Melanoma (MEL)', 'description': 'Most dangerous skin cancer. Can spread rapidly. Requires immediate medical attention.', 'risk': 'Critical', 'color': '#8B0000'},
    'nv': {'full_name': 'Melanocytic Nevi (NV)', 'description': 'Common moles. Generally benign but should be monitored for changes.', 'risk': 'Low', 'color': '#98FB98'},
    'scc': {'full_name': 'Squamous Cell Carcinoma (SCC)', 'description': 'Second most common skin cancer. Can spread if untreated. Requires treatment.', 'risk': 'High', 'color': '#FF6347'},
    'vasc': {'full_name': 'Vascular Lesions (VASC)', 'description': 'Blood vessel abnormalities. Usually benign (e.g., cherry angiomas, hemangiomas).', 'risk': 'Low', 'color': '#DDA0DD'}
}

# -------------------------
# Custom CSS (with Professional Heatmap Reveal Animation)
# -------------------------
def set_theme(background_color='#0E1117'):
    css = f"""
    <style>
    .stApp {{ background-color: {background_color}; background-image: none; }}
    .main .block-container {{ background-color: rgba(18, 18, 18, 0.8); padding: 4rem; border-radius: 12px; }}
    h1, h2, h3, h4, .stMarkdown, .stText, label, p {{ color: #F0F2F6 !important; }}
    [data-testid="stSidebar"] {{ background-color: rgba(30, 30, 30, 0.95); color: #F0F2F6; }}
    hr {{ border-top: 1px solid #333; }}

    /* Professional Heatmap Reveal Animation (Fixed) */
    .gradcam-container {{
        position: relative;
        margin-top: 1.5rem;
    }}
    .gradcam-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        opacity: 0;
        animation: gradcamFadeIn 0.3s ease-in 2.5s forwards;
    }}
    .gradcam-reveal-mask {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.12);
        clip-path: polygon(0% 0%, 0% 100%, 0% 100%, 0% 0%);
        animation: gradcamReveal 2.2s ease-in-out 0.3s forwards;
    }}
    @keyframes gradcamReveal {{
        0% {{ clip-path: polygon(0% 0%, 0% 100%, 0% 100%, 0% 0%); }}
        100% {{ clip-path: polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%); }}
    }}
    @keyframes gradcamFadeIn {{
        to {{ opacity: 1; }}
    }}
    .gradcam-caption {{
        text-align: center;
        margin-top: 10px;
        font-size: 0.95em;
        color: #aaa;
        font-style: italic;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_model():
    try:
        with st.spinner("Downloading model (this may take a minute on first run)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
        checkpoint = torch.load(BytesIO(response.content), map_location='cpu')
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------
# Grad-CAM Lite (Pure PyTorch — No Extra Deps!)
# -------------------------
def generate_gradcam_lite(model, image_tensor, predicted_class):
    """CPU-safe Grad-CAM for Streamlit Cloud."""
    try:
        # EfficientNet-B4 last conv layer
        target_layer = "blocks.6.1.conv_pwl"
        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations['act'] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients['grad'] = grad_out[0].detach()

        layer = dict(model.named_modules())[target_layer]
        fh = layer.register_forward_hook(forward_hook)
        bh = layer.register_full_backward_hook(backward_hook)

        # Forward + backward
        outputs = model(image_tensor)
        score = outputs[0, predicted_class]
        score.backward()

        # Compute CAM
        A = activations['act'].squeeze()
        G = gradients['grad'].squeeze()
        weights = torch.mean(G, dim=(1, 2))
        cam = torch.zeros(A.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * A[i]
        cam = torch.relu(cam).numpy()

        # Normalize & resize
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # Overlay
        img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        img = np.array(img)
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        fh.remove()
        bh.remove()
        return Image.fromarray(overlay)
    except Exception as e:
        return None

# -------------------------
# Image Preprocessing & Prediction
# -------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.05)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return get_transform()(image).unsqueeze(0)

def predict_with_tta(model, image_tensor, use_tta=True):
    with torch.no_grad():
        if use_tta:
            probs = torch.stack([
                F.softmax(model(image_tensor), dim=1),
                F.softmax(model(torch.flip(image_tensor, dims=[3])), dim=1),
                F.softmax(model(torch.flip(image_tensor, dims=[2])), dim=1)
            ]).mean(0)
        else:
            probs = F.softmax(model(image_tensor), dim=1)
    return probs.cpu().numpy()[0]

# -------------------------
# Visualization Utilities
# -------------------------
def create_probability_chart(probabilities, class_names):
    pairs = sorted(zip(probabilities, class_names), reverse=True)
    sorted_probs, sorted_names = zip(*pairs)
    full_names = [CLASS_INFO[n]['full_name'] for n in sorted_names]
    colors = [CLASS_INFO[n]['color'] for n in sorted_names]

    fig = go.Figure(go.Bar(
        x=[p*100 for p in sorted_probs],
        y=full_names,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p*100:.1f}%' for p in sorted_probs],
        textposition='outside'
    ))
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Lesion Type",
        height=400,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(18, 18, 18, 0.1)',
        font=dict(color='#F0F2F6'),
        xaxis=dict(range=[0, 105])
    )
    return fig
    
def create_risk_indicator(top_class: str):
    risk = CLASS_INFO[top_class]['risk']
    risk_colors = {
        'Low': '#4CAF50', 
        'Medium': '#FFC107',
        'High': '#FF5722',
        'Critical': '#F44336'
    }
    color = risk_colors.get(risk, '#808080')
    html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center; margin-bottom: 20px;">
        <h2 style="margin: 0; color: white !important;">Risk Level: {risk}</h2>
    </div>
    """
    return html, risk

# -------------------------
# Streamlit UI (✅ st.set_page_config is FIRST!)
# -------------------------
def main():
    st.set_page_config(
        page_title="Skin Scanner AI Tool",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_theme()

    st.markdown("""
        # 🔬 Skin Scanner Dermoscopic Photo Analyser
        <p style='font-size: 18px; color: #aaa; margin-top: -10px;'>
        8-Class Dermoscopic Image Classification | EfficientNet-B4 trained on 25,000 images (ISIC2019) | Macro F1 84.5% | Macro AUC 98.4% | Balanced Accuracy 83.6%
        </p>
        <hr>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("Skin Scanner analyses dermoscopic images of skin lesions, predicts the risk of malignancy and classifies the lesions into **8 categories**.")
        st.divider()
        st.warning("⚠️ **Medical Disclaimer**\n\nThis tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.")
        st.divider()
        st.header("⚙️ Settings")
        use_tta = st.checkbox("Use Test-Time Augmentation", value=True)
        show_all_probabilities = st.checkbox("Show detailed probability chart", value=True)
        show_gradcam = st.checkbox("Show AI Heatmap (Grad-CAM)", value=True)
        st.divider()
        st.header("📊 Model Performance (ISIC2019)")
        st.metric("Macro F1 Score", "0.845")
        st.metric("Macro AUC", "0.984")
        st.metric("Balanced Accuracy", "0.836")
        

    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    st.subheader("📤 Upload Dermoscopic Image")
    uploaded_file = st.file_uploader("Choose a dermoscopic image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            col1, col2 = st.columns([1, 1])
            image = Image.open(uploaded_file)

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
                st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")

                # ✅ Professional Heatmap Reveal Animation
                if show_gradcam:
                    with st.spinner("Generating AI attention map..."):
                        tensor = preprocess_image(image)
                        top_idx = np.argmax(predict_with_tta(model, tensor, use_tta=False))
                        gradcam_img = generate_gradcam_lite(model, tensor, top_idx)
                    
                    if gradcam_img:
                        # Convert image to base64 for overlay
                        buffered = BytesIO()
                        gradcam_img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        st.markdown(f"""
                        <div class="gradcam-container">
                            <img src="data:image/png;base64,{img_str}" 
                                 style="width:100%; height:auto; display:block; border-radius: 8px;"
                                 alt="Dermoscopic image with AI heatmap">
                            <div class="gradcam-overlay">
                                <div class="gradcam-reveal"></div>
                            </div>
                            <div class="gradcam-caption">
                                AI Heatmap Imaging
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("AI attention map unavailable for this image.")

            with col2:
                st.subheader("Classification Results")
                with st.spinner("Analyzing image..."):
                    tensor = preprocess_image(image)
                    probs = predict_with_tta(model, tensor, use_tta=use_tta)
                top_idx = np.argmax(probs)
                top_class = CLASS_NAMES[top_idx]

                risk_html, risk_level = create_risk_indicator(top_class)
                st.markdown(risk_html, unsafe_allow_html=True)
                # ✅ Add urgent banner for High/Critical
                if risk_level in ['High', 'Critical']:
                    st.warning("🚨 **Seek urgent Dermatology opinion** ", icon="⚠️")
                
                st.markdown("---")
                st.markdown(f"### **Predicted Diagnosis:**\n## {CLASS_INFO[top_class]['full_name']}")
                st.markdown(f"**Confidence:** <span style='font-size: 1.2em; color: #00FF7F;'>{probs[top_idx]*100:.1f}%</span>", unsafe_allow_html=True)
                st.progress(float(probs[top_idx]))
                st.markdown("---")
                st.markdown(f"**Description:** {CLASS_INFO[top_class]['description']}")

            if show_all_probabilities:
                st.subheader("📊 Detailed Probability Distribution")
                st.plotly_chart(create_probability_chart(probs, CLASS_NAMES), use_container_width=True)

            st.subheader("🩺 Clinical Recommendations")
            if risk_level in ['Critical', 'High']:
                st.error(f"**⚠️ URGENT: This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**\n\n**Recommended Actions:**\n- Schedule an appointment with a **dermatologist immediately**\n- Do not delay - early detection is crucial\n- Bring this analysis to your appointment\n- Avoid sun exposure until evaluated")
            elif risk_level == 'Medium':
                st.warning(f"**⚡ This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**\n\n**Recommended Actions:**\n- Schedule a dermatologist appointment within **1-2 weeks**\n- Monitor for any changes in size, color, or shape\n- Protect from sun exposure")
            else:
                st.info(f"**✓ This lesion appears to be {CLASS_INFO[top_class]['full_name']}**\n\n**Recommended Actions:**\n- Continue regular skin monitoring\n- Annual dermatology check-ups recommended\n- Report any changes to your doctor\n- Practice sun safety")

            st.subheader("🔍 Top 3 Predictions")
            top3 = np.argsort(probs)[::-1][:3]
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                name = CLASS_NAMES[idx]
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border: 2px solid {CLASS_INFO[name]['color']};">
                        <h4>#{i+1}: {CLASS_INFO[name]['full_name']}</h4>
                        <p><strong>Confidence:</strong> {probs[idx]*100:.1f}%</p>
                        <p><strong>Risk:</strong> {CLASS_INFO[name]['risk']}</p>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred while processing the image: {e}")
    else:
        st.info("👆 **Please upload a dermoscopic image to begin analysis**\n\n**Tips for best results:** Use high-quality dermoscopic images with good lighting and focus. Not validated for subungal or mucousal lesions.")

    st.subheader("📸 What is a dermoscopic image?")
    st.markdown("Dermoscopic images are captured using a **dermatoscope**, a specialized tool that uses magnification and polarized light to examine skin patterns beneath the surface, enabling more accurate diagnoses.")
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 20px;">
        <p><strong>Model:</strong> EfficientNet-B4 | Trained on 25,331 ISIC2019 images | 8-class classification</p>
        <p><strong>Developed by:</strong> Dr Tom Hutchinson, Oxford, England | For educational and research purposes only</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
