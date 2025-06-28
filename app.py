import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# --- Load Pipelines ---
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sentiment_analyzer = pipeline("sentiment-analysis")
    image_generator = pipeline("text-to-image", model="CompVis/stable-diffusion-v1-4")  # fallback model
    return summarizer, sentiment_analyzer, image_generator

summarizer, sentiment_analyzer, image_generator = load_pipelines()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Memory Maker", layout="wide")
st.title("🧠 AI Memory Maker")
st.write("Turn raw chat text into a visual diary using AI ✨")

# --- User Input ---
chat_input = st.text_area("📥 Paste your conversation text here:", height=250)

if st.button("✨ Generate Memory"):
    if chat_input.strip() == "":
        st.warning("Please paste some chat or conversation text.")
    else:
        with st.spinner("Analyzing and generating..."):

            # --- Step 1: Summarization ---
            summary_text = summarizer(chat_input, max_length=100, min_length=25, do_sample=False)[0]['summary_text']

            # --- Step 2: Sentiment Analysis ---
            sentiment_result = sentiment_analyzer(summary_text)[0]
            sentiment_label = sentiment_result['label']
            sentiment_score = round(sentiment_result['score'], 2)

            # --- Step 3: Image Generation ---
            try:
                image_output = image_generator(summary_text)[0]
                if 'image' in image_output:
                    image = image_output['image']
                else:
                    # Fallback to another way to get image from URL
                    image_url = image_output.get("url")
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
            except Exception as e:
                image = None
                st.error(f"Image generation failed: {e}")

            # --- Display Output ---
            st.subheader("📝 Summary")
            st.success(summary_text)

            st.subheader("💬 Sentiment")
            st.info(f"**{sentiment_label}** (Confidence: {sentiment_score})")

            if image:
                st.subheader("🎨 Visual Memory")
                st.image(image, use_column_width=True, caption="AI-generated image from summary")
