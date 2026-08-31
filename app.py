import os
import streamlit as st
from PIL import Image
import urllib.request
import torch
from torchvision import models, transforms
from ultralytics import YOLO

st.set_page_config(page_title="Smart Waste Classification", layout="wide")
st.title("Smart Waste Classification System")
st.write("Upload a photo. The app detects waste, recognizes the environment, and gives disposal advice.")

@st.cache_resource
def load_waste_model():
    return YOLO("best.pt")

@st.cache_resource
def load_environment_model():
    cat_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
    weight_url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"

    if not os.path.exists("categories_places365.txt"):
        urllib.request.urlretrieve(cat_url, "categories_places365.txt")

    if not os.path.exists("resnet18_places365.pth.tar"):
        urllib.request.urlretrieve(weight_url, "resnet18_places365.pth.tar")

    classes = []
    with open("categories_places365.txt") as f:
        for line in f:
            classes.append(line.strip().split(" ")[0][3:])

    env_model = models.resnet18(num_classes=365)
    checkpoint = torch.load("resnet18_places365.pth.tar", map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    env_model.load_state_dict(state_dict)
    env_model.eval()
    return env_model, classes

def map_environment(place_names):
    if isinstance(place_names, str):
        place_names = [place_names]

    joined = " ".join(p.lower() for p in place_names)

    if any(x in joined for x in ["beach", "coast", "ocean", "sandbar", "boardwalk", "seashore"]):
        return "beach"
    if any(x in joined for x in ["kitchen", "pantry", "bedroom", "living_room", "bathroom", "apartment", "house", "home", "dining_room"]):
        return "home"
    if any(x in joined for x in ["alley", "park", "forest", "garden", "field", "playground", "campsite"]):
        return "park"
    if any(x in joined for x in ["street", "highway", "road", "downtown", "parking", "crosswalk"]):
        return "street"
    if any(x in joined for x in ["market", "bazaar", "supermarket", "shopping", "store", "shop"]):
        return "market"
    if any(x in joined for x in ["factory", "industrial", "construction", "warehouse", "chemistry_lab"]):
        return "industrial"
    return "other"


def predict_environment(image, env_model, classes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = env_model(x)
        probs = torch.softmax(output, dim=1)[0]
        top5 = torch.topk(probs, 5)

    raw_places = [classes[int(i)] for i in top5.indices]
    environment = map_environment(raw_places)
    raw = ", ".join(raw_places[:3])
    return raw, environment

def get_recommendations(waste_items, environment):
    if not waste_items:
        return ["No waste item detected. Try another image."]
    recs = []
    for item in waste_items:
        item = item.lower()
        if item == "plastic" and environment == "beach":
            recs.append("Plastic on beach: High priority. Collect it so it does not enter water.")
        elif item == "plastic" and environment == "market":
            recs.append("Plastic in market: Put it in dry waste / recycling. Do not mix with food waste.")
        elif item == "plastic":
            recs.append(f"Plastic in {environment}: Use the dry waste / recycling bin.")
        elif item == "organic" and environment == "home":
            recs.append("Organic waste at home: Compost if possible, or use wet waste bin.")
        elif item == "organic":
            recs.append("Organic waste: Use wet waste bin. Do not mix with plastic or metal.")
        elif item in ["paper", "cardboard"]:
            recs.append(f"{item.capitalize()} in {environment}: Keep dry and put in paper recycling.")
        elif item in ["metal", "glass"]:
            recs.append(f"{item.capitalize()} in {environment}: Recyclable. Use dry waste / recycling bin.")
        else:
            recs.append(f"{item.capitalize()} in {environment}: Dispose in the correct bin.")
    return recs

waste_model = load_waste_model()
env_model, env_classes = load_environment_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or take a photo")

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif camera is not None:
    image = Image.open(camera)

if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    image.convert("RGB").save("temp.jpg")
    results = waste_model.predict("temp.jpg", conf=0.35, verbose=False)
    result_img = results[0].plot()
    result_img = Image.fromarray(result_img[..., ::-1])

    waste_items = []
    for box in results[0].boxes:
        name = results[0].names[int(box.cls[0])]
        waste_items.append(name)
    waste_items = list(set(waste_items))

    raw_place, environment = predict_environment(image, env_model, env_classes)
    recs = get_recommendations(waste_items, environment)

    with col2:
        st.subheader("Detection result")
        st.image(result_img, use_container_width=True)

    st.subheader("Results")
    st.write(f"**Environment:** {environment} ({raw_place})")
    st.write(f"**Detected waste:** {', '.join(waste_items) if waste_items else 'None'}")
    st.subheader("Recommendations")
    for r in recs:
        st.write("- " + r)