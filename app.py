"""
Skin Lesion Classification App
8-Class Dermoscopic Image Classifier using ISIC2019-trained EfficientNet-B4
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import requests
from io import BytesIO
import numpy as np
import plotly.graph_objects as go

### GRAD-CAM INTEGRATION: Import torchcam
try:
    from torchcam.methods import SmoothGradCAMpp
    from torchcam.utils import overlay_mask
    TORCHCAM_AVAILABLE = True
except ImportError:
    TORCHCAM_AVAILABLE = False
    st.warning("⚠️ Grad-CAM not available. Install 'torchcam' for AI attention maps.")

# -------------------------
# Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/Skindoc/streamlit5/resolve/main/best_model_20251116_151842.pth"
MODEL_NAME = "tf_efficientnet_b4"
NUM_CLASSES = 8
IMG_SIZE = 384

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

CLASS_INFO = {
    'akiec': {
        'full_name': 'Actinic Keratoses (AKIEC)',
        'description': 'Pre-cancerous lesions caused by sun damage. Requires monitoring and treatment.',
        'risk': 'Medium',
        'color': '#FFA500'  # Orange
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma (BCC)',
        'description': 'Most common skin cancer. Slow-growing, rarely spreads, highly treatable.',
        'risk': 'High',
        'color': '#FF4444'  # Bright Red
    },
    'bkl': {
        'full_name': 'Benign Keratosis (BKL)',
        'description': 'Non-cancerous skin growth. Generally harmless but may be removed for cosmetic reasons.',
        'risk': 'Low',
        'color': '#90EE90'  # Light Green
    },
    'df': {
        'full_name': 'Dermatofibroma (DF)',
        'description': 'Benign fibrous nodule. Usually harmless and does not require treatment.',
        'risk': 'Low',
        'color': '#87CEEB'  # Sky Blue
    },
    'mel': {
        'full_name': 'Melanoma (MEL)',
        'description': 'Most dangerous skin cancer. Can spread rapidly. Requires immediate medical attention.',
        'risk': 'Critical',
        'color': '#8B0000'  # Dark Red/Maroon
    },
    'nv': {
        'full_name': 'Melanocytic Nevi (NV)',
        'description': 'Common moles. Generally benign but should be monitored for changes.',
        'risk': 'Low',
        'color': '#98FB98'  # Pale Green
    },
    'scc': {
        'full_name': 'Squamous Cell Carcinoma (SCC)',
        'description': 'Second most common skin cancer. Can spread if untreated. Requires treatment.',
        'risk': 'High',
        'color': '#FF6347'  # Tomato Red
    },
    'vasc': {
        'full_name': 'Vascular Lesions (VASC)',
        'description': 'Blood vessel abnormalities. Usually benign (e.g., cherry angiomas, hemangiomas).',
        'risk': 'Low',
        'color': '#DDA0DD'  # Plum
    }
}

# -------------------------
# Custom CSS for Professional Look + GRAD-CAM ANIMATION
# -------------------------

def set_theme(background_color='#0E1117'):
    """Sets a consistent dark-themed style with Grad-CAM fade-in animation."""
    css = f"""
    <style>
    /* 1. Global Background Color */
    .stApp {{
        background-color: {background_color};
        background-image: none;
    }}
    
    /* 2. Main Content Container */
    .main .block-container {{
        background-color: rgba(18, 18, 18, 0.8);
        padding-top: 4rem;
        padding-right: 4rem;
        padding-left: 4rem;
        padding-bottom: 4rem;
        border-radius: 12px;
    }}
    
    /* 3. Text and Header Colors */
    h1, h2, h3, h4, .stMarkdown, .stText, label, p, .css-1456l0p, .css-1dp5vir {{
        color: #F0F2F6 !important; 
    }}
    
    /* 4. Sidebar Contrast */
    [data-testid="stSidebar"] {{
        background-color: rgba(30, 30, 30, 0.95);
        color: #F0F2F6;
    }}
    
    /* 5. Horizontal Rule */
    hr {{
        border-top: 1px solid #333;
    }}

    /* 6. GRAD-CAM ANIMATION: Smooth fade-in */
    .gradcam-container {{
        opacity: 0;
        animation: fadeInGradCAM 0.8s ease-in forwards;
        margin-top: 1.5rem;
    }}
    @keyframes fadeInGradCAM {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_model():
    """Load the trained model from HuggingFace"""
    try:
        with st.spinner("Downloading model (this may take a minute on first run)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()

        checkpoint = torch.load(BytesIO(response.content), map_location='cpu')
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

### GRAD-CAM INTEGRATION: Auto-detect target layer for timm models
def find_last_conv_layer(model, model_name):
    """Auto-find the last conv layer name for common timm models."""
    if "efficientnet" in model_name.lower():
        # EfficientNet: last pw conv in final block
        return "blocks.6.1.conv_pwl"  # B4 has 7 blocks (0-6), last is 6
    elif "convnext" in model_name.lower():
        return "stages.3.blocks.2.conv_dw"
    else:
        # Fallback: search for last 'conv' layer
        conv_layers = [name for name, _ in model.named_modules() if "conv" in name and "bn" not in name]
        return conv_layers[-1] if conv_layers else None

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
    transform = get_transform()
    tensor = transform(image).unsqueeze(0) 
    return tensor

def predict_with_tta(model: torch.nn.Module, image_tensor: torch.Tensor, use_tta: bool = True) -> np.ndarray:
    with torch.no_grad():
        if use_tta:
            probs_list = [
                F.softmax(model(image_tensor), dim=1),
                F.softmax(model(torch.flip(image_tensor, dims=[3])), dim=1),
                F.softmax(model(torch.flip(image_tensor, dims=[2])), dim=1)
            ]
            probs = torch.stack(probs_list).mean(0)
        else:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()[0]

# -------------------------
# Visualization Utilities
# -------------------------
def create_probability_chart(probabilities: np.ndarray, class_names: list) -> go.Figure:
    prob_class_pairs = list(zip(probabilities, class_names))
    prob_class_pairs.sort(key=lambda x: x[0], reverse=True)

    sorted_probs = [pair[0] for pair in prob_class_pairs]
    sorted_names = [pair[1] for pair in prob_class_pairs]

    sorted_full_names = [CLASS_INFO[name]['full_name'] for name in sorted_names]
    sorted_colors = [CLASS_INFO[name]['color'] for name in sorted_names]

    fig = go.Figure(data=[
        go.Bar(
            x=[p * 100 for p in sorted_probs],
            y=sorted_full_names,
            orientation='h',
            marker=dict(color=sorted_colors),
            text=[f'{p*100:.1f}%' for p in sorted_probs],
            textposition='outside',
        )
    ])

    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Lesion Type",
        height=400,
        showlegend=False,
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

### GRAD-CAM INTEGRATION: Generate heatmap
def generate_gradcam(model, image_tensor, predicted_class, model_name):
    """Generate Grad-CAM heatmap overlay."""
    try:
        # Auto-detect layer
        target_layer = find_last_conv_layer(model, model_name)
        if not target_layer:
            return None

        # Extract CAM
        cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
        with torch.no_grad():
            outputs = model(image_tensor)
        activation_map = cam_extractor(outputs.squeeze(0), class_idx=predicted_class)[0]

        # Resize & overlay
        original_img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        heatmap = Image.fromarray((activation_map.numpy() * 255).astype(np.uint8))
        result = overlay_mask(original_img, heatmap, alpha=0.55)

        return result
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM failed (layer='{target_layer}'): {e}")
        return None

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(
        page_title="Skin Scanner AI Tool",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_theme()

    st.markdown(
        """
        # 🔬 Skin Scanner Dermoscopic Photo Analyser
        <p style='font-size: 18px; color: #aaa; margin-top: -10px;'>
        8-Class Dermoscopic Image Classification | EfficientNet-B4 trained on 25,000 images (ISIC2019) | Macro F1 84.5% | Macro AUC 98.4% | Balanced Accuracy 83.6%
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        This AI model classifies medical dermoscopic images of skin lesions into **8 categories**.
        """)

        st.subheader("Classification Categories")
        st.markdown("""
        - **Critical/High Risk:** Melanoma, BCC, SCC
        - **Medium Risk:** Actinic Keratosis
        - **Low Risk:** Naevus, Benign Keratosis, Dermatofibroma, Vascular
        """)

        st.divider()

        st.header("⚙️ Settings")
        use_tta = st.checkbox("Use Test-Time Augmentation", value=True, 
                              help="Improves accuracy but takes slightly longer")
        show_all_probabilities = st.checkbox("Show detailed probability chart", value=True)
        
        ### GRAD-CAM INTEGRATION: Toggle in sidebar
        show_gradcam = st.checkbox(
            "Show AI Attention (Grad-CAM)",
            value=False,
            help="Highlights regions the AI used to make its decision",
            disabled=not TORCHCAM_AVAILABLE
        )
        if not TORCHCAM_AVAILABLE:
            st.caption("→ Install 'torchcam' to enable")

        st.divider()

        st.header("📊 Model Performance (ISIC2019)")
        st.metric("Macro F1 Score", "0.845")
        st.metric("Macro AUC", "0.984")
        st.metric("Balanced Accuracy", "0.836")

        st.divider()

        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This tool is for educational and research purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist.
        """)

    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    st.subheader("📤 Upload Dermoscopic Image")

    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a high-quality dermoscopic image for classification"
    )

    if uploaded_file is not None:
        try:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")

                ### GRAD-CAM INTEGRATION: Show only if enabled
                if show_gradcam and TORCHCAM_AVAILABLE:
                    st.markdown('<div class="gradcam-container">', unsafe_allow_html=True)
                    with st.spinner("Generating AI attention map..."):
                        image_tensor = preprocess_image(image)
                        top_idx = np.argmax(predict_with_tta(model, image_tensor, use_tta=False))
                        gradcam_img = generate_gradcam(model, image_tensor, top_idx, MODEL_NAME)
                        if gradcam_img:
                            st.image(gradcam_img, caption="AI Focus: Regions most influential for diagnosis", use_column_width=True)
                        else:
                            st.info("AI attention map unavailable for this image.")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.subheader("Classification Results")

                with st.spinner("Analyzing image..."):
                    image_tensor = preprocess_image(image)
                    probabilities = predict_with_tta(model, image_tensor, use_tta=use_tta)

                top_idx = np.argmax(probabilities)
                top_class = CLASS_NAMES[top_idx]
                top_prob = probabilities[top_idx]

                risk_html, risk_level = create_risk_indicator(top_class)
                st.markdown(risk_html, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"### **Predicted Diagnosis:**")
                st.markdown(f"## {CLASS_INFO[top_class]['full_name']}")
                st.markdown(f"**Confidence:** <span style='font-size: 1.2em; color: #00FF7F;'>{top_prob*100:.1f}%</span>", unsafe_allow_html=True)
                st.progress(float(top_prob))
                st.markdown("---")
                st.markdown(f"**Description:** {CLASS_INFO[top_class]['description']}")

            if show_all_probabilities:
                st.subheader("📊 Detailed Probability Distribution")
                fig = create_probability_chart(probabilities, CLASS_NAMES)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("🩺 Clinical Recommendations")
            if risk_level in ['Critical', 'High']:
                st.error(f"""
                **⚠️ URGENT: This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**
                
                **Recommended Actions:**
                - Schedule an appointment with a **dermatologist immediately**
                - Do not delay - early detection is crucial
                - Bring this analysis to your appointment
                - Consider getting a biopsy if recommended by your doctor
                """)
            elif risk_level == 'Medium':
                st.warning(f"""
                **⚡ This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**
                
                **Recommended Actions:**
                - Schedule a dermatologist appointment within **1-2 weeks**
                - Monitor for any changes in size, color, or shape
                - Consider treatment options with your doctor
                - Protect from sun exposure
                """)
            else:
                st.info(f"""
                **✓ This lesion appears to be {CLASS_INFO[top_class]['full_name']}**
                
                **Recommended Actions:**
                - Continue regular skin monitoring
                - Annual dermatology check-ups recommended
                - Report any changes to your doctor
                - Practice sun safety
                """)

            st.subheader("🔍 Top 3 Predictions")
            top_3_idx = np.argsort(probabilities)[::-1][:3]
            cols = st.columns(3)
            for i, idx in enumerate(top_3_idx):
                class_name = CLASS_NAMES[idx]
                prob = probabilities[idx]
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border: 2px solid {CLASS_INFO[class_name]['color']};">
                        <h4>#{i+1}: {CLASS_INFO[class_name]['full_name']}</h4>
                        <p><strong>Confidence:</strong> {prob*100:.1f}%</p>
                        <p><strong>Risk:</strong> {CLASS_INFO[class_name]['risk']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred while processing the image.")
            st.error(f"Error details: {str(e)}")
            st.info("Please ensure the image is a valid JPG/PNG file and try again.")

    else:
        st.info("""
        👆 **Please upload a dermoscopic image to begin analysis**
        
        **Tips for best results:** Use high-quality dermoscopic images with good lighting and focus. Not validated for subungal or mucousal lesions. 
        """)

    st.subheader("📸 What is a dermoscopic image?")
    st.markdown("""
    Dermoscopic images are captured using a **dermatoscope**, a specialized tool that uses magnification and polarized light to examine skin patterns beneath the surface, enabling more accurate diagnoses.
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 20px;">
        <p><strong>Model:</strong> EfficientNet-B4 | Trained on 25,331 ISIC2019 images | 8-class classification</p>
        <p><strong>Developed by:</strong> Dr Tom Hutchinson, Oxford, England | For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
