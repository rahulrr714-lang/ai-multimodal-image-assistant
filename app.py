import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="AI Image Description App")

st.title("🖼️ AI Image Description App")
st.write("Upload an image and get an AI-generated description.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load model (cached so it loads only once)
    @st.cache_resource
    def load_model():
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        return processor, model

    processor, model = load_model()

    # Generate caption
    with st.spinner("🧠 Generating AI description..."):
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success("✅ AI Description Generated")
    st.write("📝 **Description:**")
    st.write(caption)

