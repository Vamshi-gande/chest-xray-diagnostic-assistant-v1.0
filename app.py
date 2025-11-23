"""
Chest X-ray Diagnostic Assistant - Hugging Face Spaces
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
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Chest X-ray Diagnostic Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
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
    device = torch.device('cpu')  # Use CPU for Hugging Face free tier
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
    
    # Storage for activations and gradients
    activation_storage = {'value': None}
    gradient_storage = {'value': None}
    
    def forward_hook(module, input, output):
        # Store the activation
        activation_storage['value'] = output.detach()
        
        # Register a hook on the output tensor to capture gradients
        def backward_hook(grad):
            gradient_storage['value'] = grad.detach()
            return None  # Don't modify the gradient
        
        # Register hook on the output tensor
        output.register_hook(backward_hook)
    
    # Register forward hook
    handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        # Forward pass with fresh tensor
        input_tensor = image_tensor.detach().clone()
        input_tensor.requires_grad = True
        
        output = model(input_tensor)
        max_idx = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        target_score = output[0, max_idx]
        target_score.backward()
        
        # Remove hook
        handle.remove()
        
        # Check if we got the data
        if activation_storage['value'] is None or gradient_storage['value'] is None:
            raise ValueError("Failed to capture activations or gradients")
        
        # Get stored values
        gradients = gradient_storage['value']
        activations = activation_storage['value']
        
        # Move to CPU for processing
        gradients = gradients.cpu()
        activations = activations.cpu()
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)
        
        # Get the heatmap
        heatmap = weighted_activations.squeeze().numpy()
        
        # Apply ReLU and normalize
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

# Use Hugging Face Inference API (Free!)
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
        
        # Confidence calculation based on distance from threshold
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
        
        # Format for LLM
        findings_for_llm.append(f"{label}: {prob:.1%} ({status})")
    
    findings_text += "\n**IMPRESSION:**\n\n"
    
    # Try LLM first
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
            # Fallback to rule-based
            use_llm = False
    
    # Rule-based fallback
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
    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>Chest X-ray Diagnostic Assistant</h1>'
        '<p>AI-powered analysis with explainable predictions using custom thresholds</p>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **This AI system analyzes chest X-rays for:**
        - Cardiomegaly (enlarged heart)
        - Edema (fluid accumulation)
        - Consolidation (lung tissue solidification)
        - Atelectasis (lung collapse)
        - Pleural Effusion (fluid around lungs)
        
        **Technology:**
        - DenseNet-121 CNN
        - Grad-CAM visualization
        - Hugging Face LLM (Phi-3-mini)
        - Custom per-condition thresholds
        """)
        
        st.warning("⚠️ **Disclaimer**: Research prototype only. Not for clinical diagnosis.")
        
        st.markdown("---")
        
        # LLM Toggle
        st.markdown("**AI Report Generation:**")
        use_llm = st.checkbox("Use LLM for reports", value=True, 
                             help="Uses Hugging Face Inference API (Phi-3-mini) for natural language reports")
        
        if use_llm:
            has_token = hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets
            if has_token:
                st.success("✅ HF Token detected")
            else:
                st.info("💡 Add HF_TOKEN to secrets for unlimited API calls")
        
        st.markdown("---")
        st.markdown("**Model Performance:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("AUROC", "0.82")
        with col_b:
            st.metric("Accuracy", "78%")
        
        st.markdown("---")
        st.markdown("**Custom Thresholds:**")
        for label, thresh in THRESHOLDS.items():
            st.text(f"{label}: {thresh:.0%}")
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
        st.success("Model loaded successfully!")
    
    # Main area
    st.header("Upload Chest X-ray")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an X-ray image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a frontal chest X-ray for analysis"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
               
        # Display original BEFORE analysis
        st.subheader("Original X-ray")
        st.image(image, use_column_width=True)
        
        # Analyze button
        if st.button("Analyze X-ray", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                image_tensor = preprocess_image(image).to(device)
                
                # Predict
                with torch.no_grad():
                    predictions = model(image_tensor).cpu().numpy()[0]
                
                labels = list(THRESHOLDS.keys())
                
                # Generate Grad-CAM
                try:
                    cam = generate_gradcam_simple(model, image_tensor, device)
                    visualization = overlay_heatmap(image, cam)
                    
                    # NOW create columns INSIDE the button block
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        st.subheader("Original")
                        st.image(image, use_column_width=True)

                    with col2:
                        st.subheader("Attention Heatmap")
                        st.image(cam, width=400)
                    
                    with col3:
                        st.subheader("Overlay")
                        st.image(visualization, width=400)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate Grad-CAM: {e}")
                
                st.markdown("---")
                
                # Predictions with custom thresholds
                st.subheader("Prediction Results")
                
                # Create metrics in columns
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
                
                # Detailed table
                st.markdown("### Detailed Analysis")
                for label, prob in zip(labels, predictions):
                    threshold = THRESHOLDS[label]
                    status_emoji = "✅" if prob > threshold else "❌"
                    
                    # Create progress bar
                    col_label, col_bar = st.columns([1, 3])
                    with col_label:
                        st.write(f"{status_emoji} **{label}**")
                    with col_bar:
                        st.progress(float(prob))
                        st.caption(f"Probability: {prob:.1%} | Threshold: {threshold:.0%}")
                
                # Bar chart
                st.subheader("Confidence Levels")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Color bars based on threshold
                colors = ['#FF6B6B' if p > THRESHOLDS[l] else '#4ECDC4' 
                         for l, p in zip(labels, predictions)]
                bars = ax.barh(labels, predictions, color=colors)
                
                # Add threshold lines
                for idx, (label, thresh) in enumerate(THRESHOLDS.items()):
                    ax.plot([thresh, thresh], [idx-0.4, idx+0.4], 
                           'k--', linewidth=2, alpha=0.5)
                
                ax.set_xlabel('Probability', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_title('Predictions vs Custom Thresholds (dashed lines)', fontsize=14)
                
                # Add value labels
                for bar, prob in zip(bars, predictions):
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1%}', ha='left', va='center', 
                           fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Clinical report
                st.markdown("---")
                st.subheader("Clinical Report")
                report = generate_report(predictions)
                st.markdown(report)
                
                # Download button
                st.download_button(
                    label="📥 Download Report (TXT)",
                    data=report,
                    file_name="chest_xray_report.txt",
                    mime="text/plain"
                )
                
                # Additional info
                with st.expander("ℹ️ Understanding the Results"):
                    st.markdown("""
                    **Custom Threshold System:**
                    - Each condition has an optimized detection threshold
                    - Thresholds are tuned for better accuracy per condition
                    - Cardiomegaly: 28% | Edema: 32% | Consolidation: 25%
                    - Atelectasis: 30% | Pleural Effusion: 35%
                    
                    **Confidence Levels:**
                    - **High**: Probability differs from threshold by >20%
                    - **Moderate**: Probability differs by 10-20%
                    - **Low**: Probability differs by <10%
                    
                    **Grad-CAM Heatmap:**
                    - Shows which regions the AI focused on
                    - Red/yellow = high attention
                    - Blue/purple = low attention
                    
                    **Important Notes:**
                    - This is an AI research tool, not a medical device
                    - Always consult a qualified radiologist for diagnosis
                    - Results should be used with clinical findings
                    """)
    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload a chest X-ray image to begin analysis")
        
        st.markdown("### How to Use:")
        st.markdown("""
        1. **Upload** a frontal chest X-ray (JPG or PNG format)
        2. Click **"Analyze X-ray"** button
        3. View predictions with custom thresholds
        4. Examine Grad-CAM heatmaps
        5. Read the AI-generated clinical report
        6. Download results for your records
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using PyTorch, Streamlit, and Hugging Face Spaces</p>
        <p><small>Model trained on CheXpert dataset | DenseNet-121 architecture | Custom threshold optimization</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()