# Similar Image Finder

A web application that finds visually similar images from a curated dataset of Unsplash Lite photos using OpenAI's CLIP model and K-Nearest Neighbors (KNN) algorithm. Built with Streamlit and deployed at:  
[**App Link**](https://similar-image-finder-sss.streamlit.app/)

---

## What It Does

This app lets you:
- Upload an image or provide an image URL.
- Extract image embeddings using OpenAI's CLIP (ViT-B/32) model.
- Retrieve and display top 6 visually similar images from a dataset of **21,000 Unsplash Lite images** using a KNN model trained on CLIP embeddings.

---

## Under the Hood

- **CLIP Model**: Generates image embeddings (`ViT-B/32`) from user input.
- **KNN Model**: Pretrained on 21k image embeddings from the Unsplash Lite dataset using `sklearn.neighbors.NearestNeighbors`.
- **Unsplash Lite Dataset**: A downsampled and curated version of the Unsplash dataset with ~21k high-quality photo URLs and thumbnails.
- **Preprocessed**: Image embeddings were generated in advance for faster search using CLIP, and saved using `pickle`.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/similar-image-finder.git
cd similar-image-finder
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended), then install requirements:

```bash
pip install -r requirements.txt
```

> Note: CLIP will be installed directly from the GitHub source.

### 3. Run the App Locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to view the app.

---

## Project Structure

```
├── app.py                     # Main Streamlit app
├── similar_image_model.sav    # Pickled KNN model + image URL index
├── photos.tsv000              # Raw Unsplash Lite metadata (tsv)
├── SimilarImageFinder.ipynb   # Preprocessing and model training notebook
├── requirements.txt           # Required Python packages
└── README.md                  # Project description
```

---

## Packages Used

- `streamlit` – for the web UI
- `torch`, `torchvision` – for using CLIP and preprocessing
- `scikit-learn` – for KNN model
- `matplotlib`, `PIL` – for image display and processing
- `requests` – to handle image URL fetching

---

## Demo

Try the live app:  
[https://similar-image-finder-sss.streamlit.app/](https://similar-image-finder-sss.streamlit.app/)

You can either:
- Paste an image URL, or
- Upload a photo from your computer

The app will return 6 visually similar images from the Unsplash Lite collection.

---

## Course Information

This project was developed as part of **CSCI 6364 - Machine Learning** at **The George Washington University**.

---

## Authors

- **Sudharshan Reddy Thammaiahgari**
- **Sneha Uppu**
- **Sri Murari Dachepalli**

---

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Unsplash Dataset](https://unsplash.com/data)
