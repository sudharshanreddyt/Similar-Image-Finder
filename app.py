import os
import torch
import clip
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from io import StringIO
import tempfile

@st.cache_resource
def load_models():
    # Load models once and cache them
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    
    file_name = "similar_image_model.sav"
    with open(file_name, 'rb') as f:
        loaded_model = pickle.load(f)
    
    return {
        'clip_model': model,
        'knn_model': loaded_model['knn_model'],
        'image_thumbnails': loaded_model['image_thumbnails'],
        'device': device
    }


def load_image_with_loaded_model(input_source, models, is_url=True):

    model_loaded = models['clip_model']
    knn_loaded = models['knn_model']
    image_thumbnails_loaded = models['image_thumbnails']
    device = models['device']

    similar_image_urls = []

    if is_url:
        own_img = Image.open(BytesIO(requests.get(input_source).content)).convert("RGB")
    else:
        own_img = Image.open(input_source).convert("RGB")    


    # Preprocess image using CLIP's transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = preprocess(own_img).unsqueeze(0).to(device)

    # Get CLIP embedding
    with torch.no_grad():
        own_embedding = model_loaded.encode_image(input_tensor)
        own_embedding = own_embedding / own_embedding.norm(dim=-1, keepdim=True)
    own_embedding_np = own_embedding.cpu().numpy()

    # Find top 30 candidates, pick first 6 valid images
    _, all_indices = knn_loaded.kneighbors(own_embedding_np, n_neighbors=30)

    match_count = 0

    for idx in all_indices[0]:
        if image_thumbnails_loaded[idx] is None:
            continue
        try:
            # Lazy load from URL
            response = requests.get(image_thumbnails_loaded[idx], timeout=5)
            if response.status_code == 200:
                similar_image_urls.append(image_thumbnails_loaded[idx])
                match_count += 1
    
            if match_count == 6:
                break
        except:
            continue  # skip broken image URLs

    return similar_image_urls


# Main function
def main():

    # Giving a title
    st.title("Similar Image Finder")

    models = load_models()

    # Initialize input_image
    input_image = None
    input_source = None
    is_url = True

    # Getting input url from the user
    url_input = st.text_input("Enter the image url")
    if url_input:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url_input,headers, timeout= 10)
            input_image = Image.open(BytesIO(response.content))
            input_source = url_input
            is_url = True
        except:
            st.error("Could not load the image")

    # Upload an image button
    uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_source = tmp_file.name
            print("Url is",input_source)

        is_url = False

    if input_image:
        st.markdown("Input Image")
        st.image(input_image, width=400)

    if(st.button("Find Similar Images")):
        if not input_source:
            st.warning("Enter either image url or upload a file to generate similar images")
            return

        # code for prediction
        with st.spinner("Finding similar images..."):
            similar_image_urls = load_image_with_loaded_model(input_source, models, is_url)

        if similar_image_urls:
            st.success("Top 6 Similar Images are : ")
            row1_cols = st.columns(3)  
            row2_cols = st.columns(3)  

            for i, url in enumerate(similar_image_urls):
                if i < 3:
                    with row1_cols[i]:
                        st.markdown(f"Image {i+1}")
                        st.image(url, width = 400)
                else:
                    with row2_cols[i-3]:
                        st.markdown(f"Image {i+1}")
                        st.image(url, width = 400)
        else:
            st.warning("No Similar Images found")


# Invoke this main function only
if __name__ == "__main__":
    main()
