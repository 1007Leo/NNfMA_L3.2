import requests
import streamlit as st
from PIL import Image, ImageDraw

BENTO_SERVICE_URL = "http://bento:3000/predict"

def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        box = pred["box"]
        name = pred["name"]
        confidence = pred["confidence"]

        draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline="red", width=3)
        
        label = f"{name}: {confidence:.2f}"
        draw.text((box["x1"], box["y1"]), label, fill="red")
    
    return image

st.title("YOLOv8 Object Detection with BentoML")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open(uploaded_file.name,"wb") as im:
        im.write(uploaded_file.getbuffer())

    response = requests.post(BENTO_SERVICE_URL, files={"images": open(uploaded_file.name, "rb")})

    if response.status_code == 200:
        predictions = response.json()
        image_with_boxes = draw_boxes(Image.open(uploaded_file), predictions)
        st.image(image_with_boxes, caption="Image with Detections", use_container_width=True)
    else:
        st.error(f"Error in prediction. Status code: {response.status_code}")
