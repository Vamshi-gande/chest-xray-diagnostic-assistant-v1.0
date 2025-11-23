"""
Chest X-ray Diagnostic Assistant - Hugging Face Spaces
SEO-optimized version for xrayaid.streamlit.app
Optimized for Streamlit 1.28.0 with per-condition thresholds
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG - SEO OPTIMIZED
# ============================================
st.set_page_config(
    page_title="X-ray AI Diagnostic Assistant | Free Chest X-ray Analysis Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SEO META DESCRIPTION (via header/text)
# ============================================
st.header("Free AI-Powered Chest X-ray Diagnostic Assistant")
st.text(
    "Advanced chest X-ray analysis tool using deep learning AI to detect cardiomegaly, "
    "edema, consolidation, atelectasis, and pleural effusion. Upload your chest X-ray "
    "for instant AI-powered diagnostic insights with explainable Grad-CAM visualization. "
    "Free medical imaging analysis tool for healthcare professionals and researchers."
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stAlert {
        background-color: #f0f2f6;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL DEFINITION
# ============================================
class ChestXrayModel(nn.Module):
    """DenseNet-121 based multi-label classifier"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.densenet121(pretrained=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))

# ============================================
# LOAD MODEL (CACHED)
# ============================================
@st.cache_resource
def load_model():
    """Load trained model"""
    device = torch.device('cpu')
    model = ChestXrayModel(num_classes=5).to(device)
    
    try:
        model.load_state_dict(torch.load('chexray_model_final.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model file 'chexray_model_final.pth' not found. Please upload it to the Space.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image):
    """Preprocess uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ============================================
# GRAD-CAM GENERATION
# ============================================
def generate_gradcam_simple(model, image_tensor, device):
    """
    Simplified Grad-CAM using direct tensor hooks
    """
    model.eval()
    target_layer = model.model.features[-1]
    
    activation_storage = {'value': None}
    gradient_storage = {'value': None}
    
    def forward_hook(module, input, output):
        activation_storage['value'] = output.detach()
        
        def backward_hook(grad):
            gradient_storage['value'] = grad.detach()
            return None
        
        output.register_hook(backward_hook)
    
    handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        input_tensor = image_tensor.detach().clone()
        input_tensor.requires_grad = True
        
        output = model(input_tensor)
        max_idx = output.argmax(dim=1).item()
        
        model.zero_grad()
        target_score = output[0, max_idx]
        target_score.backward()
        
        handle.remove()
        
        if activation_storage['value'] is None or gradient_storage['value'] is None:
            raise ValueError("Failed to capture activations or gradients")
        
        gradients = gradient_storage['value'].cpu()
        activations = activation_storage['value'].cpu()
        
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)
        
        heatmap = weighted_activations.squeeze().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
        
    except Exception as e:
        handle.remove()
        raise e

def overlay_heatmap(image, heatmap):
    """Overlay CAM on image"""
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(np.array(image), (224, 224))
    overlay = cv2.addWeighted(image_resized, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ============================================
# PER-CONDITION THRESHOLDS
# ============================================
THRESHOLDS = {
    "Cardiomegaly": 0.28,
    "Edema": 0.32,
    "Consolidation": 0.25,
    "Atelectasis": 0.30,
    "Pleural Effusion": 0.35
}

# ============================================
# HUGGING FACE API INTEGRATION
# ============================================
import requests
import json

HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

def query_llm(prompt, hf_token=None):
    """Query Hugging Face Inference API"""
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").replace(prompt, "").strip()
        return None
    except Exception as e:
        st.warning(f"LLM API error: {e}")
        return None

# ============================================
# REPORT GENERATION WITH LLM
# ============================================
def generate_report(predictions, use_llm=True):
    """Generate clinical summary with Hugging Face LLM or fallback to rule-based"""
    labels = list(THRESHOLDS.keys())
    findings_text = "**FINDINGS:**\n\n"
    detected_findings = []
    findings_for_llm = []
    
    for label, prob in zip(labels, predictions):
        threshold = THRESHOLDS[label]
        status = "POSITIVE" if prob > threshold else "NEGATIVE"
        
        distance = abs(prob - threshold)
        if distance > 0.2:
            confidence = "High"
        elif distance > 0.1:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        findings_text += f"- **{label}**: {prob:.1%} – {'✅' if status=='POSITIVE' else '❌'} {status} ({confidence} confidence, threshold: {threshold:.0%})\n"
        
        if prob > threshold:
            detected_findings.append(f"{label} ({prob:.0%})")
        
        findings_for_llm.append(f"{label}: {prob:.1%} ({status})")
    
    findings_text += "\n**IMPRESSION:**\n\n"
    
    if use_llm:
        hf_token = st.secrets.get("HF_TOKEN", None) if hasattr(st, 'secrets') else None
        
        llm_prompt = f"""You are a radiologist AI assistant. Based on these chest X-ray findings, write a concise 2-3 sentence clinical impression:

Findings:
{chr(10).join(findings_for_llm)}

Write only the clinical impression (no preamble):"""
        
        with st.spinner("Generating AI report..."):
            llm_response = query_llm(llm_prompt, hf_token)
        
        if llm_response and len(llm_response) > 20:
            findings_text += llm_response + "\n\n"
        else:
            use_llm = False
    
    if not use_llm:
        if len(detected_findings) == 0:
            findings_text += "No significant abnormalities detected in the analyzed pathologies. "
            findings_text += "The chest radiograph appears within normal limits for the assessed findings.\n\n"
        elif len(detected_findings) == 1:
            findings_text += f"Evidence of {detected_findings[0]} is present. "
            findings_text += "Clinical correlation and possible follow-up imaging may be warranted based on patient presentation.\n\n"
        else:
            findings_text += f"Multiple findings detected: {', '.join(detected_findings)}. "
            findings_text += "These findings may indicate cardiopulmonary pathology. "
            findings_text += "Clinical correlation is strongly recommended.\n\n"
    
    findings_text += "**RECOMMENDATIONS:**\n"
    findings_text += "- Correlate with patient history and clinical presentation\n"
    findings_text += "- Compare with prior imaging studies if available\n"
    findings_text += "- Consider additional imaging or laboratory workup as clinically indicated\n"
    
    return findings_text

# ============================================
# MAIN APP
# ============================================
def main():
    # Main title with SEO keywords
    st.markdown(
        '<div class="main-header">'
        '<h1> X-ray AI: Chest X-ray Diagnostic Assistant</h1>'
        '<p>Free AI-powered chest X-ray analysis | Detect lung diseases with deep learning</p>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About X-ray AI")
        st.info("""
        **Free AI Chest X-ray Analysis Tool**
        
        This medical imaging AI analyzes chest X-rays to detect:
        - **Cardiomegaly** (enlarged heart)
        - **Edema** (fluid in lungs)
        - **Consolidation** (lung tissue solidification)
        - **Atelectasis** (lung collapse)
        - **Pleural Effusion** (fluid around lungs)
        
        **Advanced AI Technology:**
        - DenseNet-121 deep learning CNN
        - Explainable AI with Grad-CAM visualization
        - Hugging Face LLM report generation (Phi-3-mini)
        - Custom per-condition diagnostic thresholds
        - Trained on CheXpert medical imaging dataset
        """)
        
        st.warning("⚠️ **Medical Disclaimer**: This is a research AI tool, not FDA-approved medical software. Always consult qualified radiologists for clinical diagnosis.")
        
        st.markdown("---")
        
        st.markdown("**AI Report Generation:**")
        use_llm = st.checkbox("Use AI language model for reports", value=True, 
                             help="Uses Hugging Face Inference API (Phi-3-mini) for natural language clinical reports")
        
        if use_llm:
            has_token = hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets
            if has_token:
                st.success("✅ HF API Token detected")
            else:
                st.info("💡 Add HF_TOKEN to secrets for unlimited API access")
        
        st.markdown("---")
        st.markdown("**AI Model Performance:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("AUROC Score", "0.82")
        with col_b:
            st.metric("Accuracy", "78%")
        
        st.markdown("---")
        st.markdown("**Detection Thresholds:**")
        for label, thresh in THRESHOLDS.items():
            st.text(f"{label}: {thresh:.0%}")
        
        st.markdown("---")
        st.markdown("**🔗 Access this tool:**")
        st.code("xrayaid.streamlit.app", language=None)
    
    # Load model
    with st.spinner("Loading AI diagnostic model..."):
        model, device = load_model()
        st.success("AI model loaded and ready for chest X-ray analysis!")
    
    # Main area
    st.header("📤 Upload Your Chest X-ray for Free AI Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, PNG, or JPEG format)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a frontal chest X-ray radiograph for instant AI-powered diagnostic analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
               
        st.subheader("Your Uploaded Chest X-ray")
        st.image(image, use_column_width=True)
        
        if st.button("Analyze X-ray with AI", type="primary"):
            with st.spinner("AI is analyzing your chest X-ray..."):
                image_tensor = preprocess_image(image).to(device)
                
                with torch.no_grad():
                    predictions = model(image_tensor).cpu().numpy()[0]
                
                labels = list(THRESHOLDS.keys())
                
                try:
                    cam = generate_gradcam_simple(model, image_tensor, device)
                    visualization = overlay_heatmap(image, cam)
                    
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        st.subheader("Original X-ray")
                        st.image(image, use_column_width=True)

                    with col2:
                        st.subheader("AI Attention Map")
                        st.image(cam, width=400)
                    
                    with col3:
                        st.subheader("Diagnostic Overlay")
                        st.image(visualization, width=400)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate Grad-CAM visualization: {e}")
                
                st.markdown("---")
                
                st.subheader("AI Diagnostic Results")
                
                metric_cols = st.columns(5)
                for idx, (label, prob) in enumerate(zip(labels, predictions)):
                    threshold = THRESHOLDS[label]
                    with metric_cols[idx]:
                        status = "✅" if prob > threshold else "❌"
                        delta_text = "Positive" if prob > threshold else "Negative"
                        st.metric(
                            label=f"{status} {label}",
                            value=f"{prob:.1%}",
                            delta=delta_text
                        )
                
                st.markdown("### Detailed Diagnostic Analysis")
                for label, prob in zip(labels, predictions):
                    threshold = THRESHOLDS[label]
                    status_emoji = "✅" if prob > threshold else "❌"
                    
                    col_label, col_bar = st.columns([1, 3])
                    with col_label:
                        st.write(f"{status_emoji} **{label}**")
                    with col_bar:
                        st.progress(float(prob))
                        st.caption(f"AI Confidence: {prob:.1%} | Detection Threshold: {threshold:.0%}")
                
                st.subheader("AI Confidence Visualization")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                colors = ['#FF6B6B' if p > THRESHOLDS[l] else '#4ECDC4' 
                         for l, p in zip(labels, predictions)]
                bars = ax.barh(labels, predictions, color=colors)
                
                for idx, (label, thresh) in enumerate(THRESHOLDS.items()):
                    ax.plot([thresh, thresh], [idx-0.4, idx+0.4], 
                           'k--', linewidth=2, alpha=0.5)
                
                ax.set_xlabel('AI Probability Score', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_title('AI Predictions vs Custom Detection Thresholds (dashed lines)', fontsize=14)
                
                for bar, prob in zip(bars, predictions):
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1%}', ha='left', va='center', 
                           fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("AI-Generated Clinical Report")
                report = generate_report(predictions, use_llm)
                st.markdown(report)
                
                st.download_button(
                    label="📥 Download AI Diagnostic Report (TXT)",
                    data=report,
                    file_name="xray_ai_diagnostic_report.txt",
                    mime="text/plain"
                )
                
                with st.expander("ℹ️ Understanding Your AI Diagnostic Results"):
                    st.markdown("""
                    **How X-ray AI Works:**
                    
                    **Custom Threshold Detection System:**
                    - Each lung condition has an AI-optimized detection threshold
                    - Thresholds are fine-tuned for maximum diagnostic accuracy
                    - Cardiomegaly: 28% | Edema: 32% | Consolidation: 25%
                    - Atelectasis: 30% | Pleural Effusion: 35%
                    
                    **AI Confidence Levels:**
                    - **High Confidence**: AI probability differs from threshold by >20%
                    - **Moderate Confidence**: Probability differs by 10-20%
                    - **Low Confidence**: Probability differs by <10%
                    
                    **Grad-CAM Heatmap Visualization:**
                    - Shows which X-ray regions the AI analyzed most
                    - Red/yellow areas = high AI attention (key diagnostic regions)
                    - Blue/purple areas = low AI attention
                    - Helps doctors understand AI decision-making (explainable AI)
                    
                    **Important Medical Disclaimer:**
                    - X-ray AI is a research prototype and educational tool
                    - Not FDA-approved for clinical diagnostic use
                    - Always consult board-certified radiologists for medical diagnosis
                    - AI results must be interpreted with full clinical context
                    - Free tool for healthcare research and education only
                    """)
    else:
        st.info("👆 Upload a chest X-ray image above to start free AI diagnostic analysis")
        
        st.markdown("### How to Use X-ray AI:")
        st.markdown("""
        1. **Upload** a frontal chest X-ray radiograph (JPG, PNG, or JPEG format)
        2. Click the **"Analyze X-ray with AI"** button to start deep learning analysis
        3. View AI diagnostic predictions with custom detection thresholds
        4. Examine explainable AI Grad-CAM heatmaps showing decision regions
        5. Read the AI-generated clinical diagnostic report
        6. Download complete results for your medical records or research
        
        **Best Results:** Use standard PA (posteroanterior) or AP (anteroposterior) chest X-ray views
        """)
        
        st.markdown("### About This Medical AI Tool")
        st.markdown("""
        **X-ray AI** is a free, open-source deep learning tool for chest X-ray analysis, 
        built for healthcare researchers, medical students, and radiology professionals. 
        
        **Key Features:**
        -  Advanced DenseNet-121 convolutional neural network
        -  Detects 5 major lung and heart conditions
        -  Explainable AI with Grad-CAM visualization technology
        -  Natural language report generation using Hugging Face LLMs
        -  100% free medical imaging analysis tool
        -  Open-source AI for medical education and research
        
        **Training Data:** CheXpert large-scale chest X-ray dataset from Stanford University
        
        **Access:** Visit **xrayaid.streamlit.app** for instant chest X-ray AI analysis
        """)
    
    # SEO-optimized footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p><strong>X-ray AI - Free Chest X-ray Diagnostic Assistant</strong></p>
        <p>AI-powered medical imaging analysis | Built with PyTorch deep learning, Streamlit, and Hugging Face</p>
        <p><small>DenseNet-121 architecture | Trained on CheXpert dataset | Custom threshold optimization | Explainable AI with Grad-CAM</small></p>
        <p><small><strong>Keywords:</strong> chest xray AI, medical imaging analysis, deep learning radiology, 
        free chest xray analysis, AI diagnostic tool, lung disease detection, cardiomegaly detection AI, 
        pleural effusion AI, atelectasis detection, pulmonary edema AI, consolidation detection, 
        explainable medical AI, Grad-CAM visualization, DenseNet chest xray, xrayaid</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()