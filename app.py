import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

from src.model import RobustAttentionGuidedEdgeViT

st.set_page_config(page_title="Cotton Leaf Disease Classifier", page_icon="🌿", layout="wide")
st.title("Attention-Guided Edge ViT Diagnostic Dashboard")
st.write("Upload an image of a cotton leaf to extract its visual edges and run structural disease diagnostics.")

@st.cache_resource
def load_model():
    model = RobustAttentionGuidedEdgeViT(num_classes=7, edge_mode='sobel', pretrained=False)
    
    model_path = "best_hybrid_vit_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        st.warning(f"Model weights not found at `{model_path}`. Please run the training script first to generate this file.")
        
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose a cotton leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits, edge_maps = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)
        prediction = torch.argmax(probabilities).item()

    edge_np = edge_maps[0].permute(1, 2, 0).numpy()
    edge_scaled = (edge_np - edge_np.min()) / (edge_np.max() - edge_np.min() + 1e-6)

    with col2:
        st.subheader("Extracted Edge Map")
        st.image(edge_scaled, use_container_width=True, clamp=True)
        
    with col3:
        st.subheader("Diagnostic Output")
        
        classes = [
            "Bacterial Blight", "Curl Virus", "Fusarium Wilt", 
            "Healthy", "Herbicide Injury", "Leaf Spot", "Spider Mites"
        ]
        
        st.metric(label="Primary Diagnosis", value=classes[prediction])
        st.write(f"Confidence Score: **{probabilities[prediction]*100:.2f}%**")
        
        with st.expander("View Full Confidence Scores"):
            for i, class_name in enumerate(classes):
                st.write(f"**{class_name}:** {probabilities[i]*100:.1f}%")
