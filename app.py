import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from models.model import PlantCNN



st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱")

st.title("🌱 Plant Disease Detection AI")
st.write("Upload a leaf image and get prediction + confidence")



@st.cache_resource
def load_model():
    model = PlantCNN()
    state = torch.load("plant_cnn.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()



class_names = [
    "Potato Early Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)

        confidence, predicted = torch.max(probs, 1)

    label = class_names[predicted.item()]
    conf = confidence.item() * 100

    

    st.subheader("Prediction")
    st.success(label)

    st.subheader("Confidence")
    st.info(f"{conf:.2f}%")

    st.subheader("Class Probabilities")

    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probs[0][i].item()*100:.2f}%")