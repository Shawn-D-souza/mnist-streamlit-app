import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps # Make sure ImageOps is imported
import numpy as np
import traceback # For detailed error messages

# Function to load the model (cached for performance)
@st.cache_resource
def load_model_cached():
    try:
        model = tf.keras.models.load_model('mnist_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_cached()

st.title("Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if model is None:
    st.info("Model not loaded. Please check logs.")
    st.stop()

if uploaded_file is not None:
    try:
        image_original = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.subheader("1. Original Uploaded Image (Grayscale)")
        # Use use_container_width instead of use_column_width
        st.image(image_original, caption='Uploaded Grayscale Image.', use_container_width=True)

        # --- Decide if initial inversion is needed ---
        # MNIST expects a white digit on a black background.
        # If your typical uploaded image is a BLACK digit on a WHITE background,
        # you should invert it. If it's already white on black, you can skip this.
        # This is a crucial step to get right based on your input.
        
        # For debugging, let's make this an option or show both:
        st.write("--- Preprocessing Steps ---")
        image_for_processing = image_original # Default to original

        # Add a checkbox to control initial inversion for easy testing
        if st.checkbox("Invert original image (if it's black digit on white background)", value=True): # Default to True as common case
            image_for_processing = ImageOps.invert(image_original)
            st.subheader("1a. Image After Potential Initial Inversion")
            st.image(image_for_processing, caption='Image after considering initial inversion.', use_container_width=True)
        else:
            st.subheader("1a. Image (No Initial Inversion)")
            st.image(image_for_processing, caption='Original grayscale image used directly.', use_container_width=True)


        # --- Strategy 1: Resizing with Aspect Ratio Preservation and Padding ---
        desired_size = 28
        old_size = image_for_processing.size # Use the (potentially inverted) image
        
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        
        img_resized_aspect = image_for_processing.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create a new_image with a black background (pixel value 0)
        final_28x28_img = Image.new("L", (desired_size, desired_size), 0)
        
        # Paste the resized (and potentially inverted) digit onto the black canvas
        paste_x = (desired_size - new_size[0]) // 2
        paste_y = (desired_size - new_size[1]) // 2
        final_28x28_img.paste(img_resized_aspect, (paste_x, paste_y))
        
        st.subheader("2. Image Resized & Padded to 28x28")
        st.caption("This is what the model will effectively 'see' before normalization.")
        st.image(final_28x28_img, caption='Processed for Model Input (should be white digit on black bg).', use_container_width=True)
        
        # Convert to numpy array
        img_array = np.array(final_28x28_img)
        st.write(f"Shape after padding: {img_array.shape}, Min/Max pixel: {img_array.min()}/{img_array.max()}, Dtype: {img_array.dtype}")
        st.write("Check above: Min pixel should be close to 0 (black background), Max pixel close to 255 (white digit).")

        # NO further global inversion (like 255 - img_array) should be needed here if the above steps
        # correctly result in a white digit on a black background in 'final_28x28_img'.

        # 3. Normalize the image (scale pixel values to 0-1)
        # The 'img_array' should now correctly represent white digit (high values) on black background (low values)
        normalized_img_array = img_array.astype("float32") / 255.0
        st.subheader("3. Normalized Image Array (Pixel values 0-1)")
        # Displaying normalized image requires it to be in a displayable range, st.image handles this.
        st.image(normalized_img_array, caption='Normalized (Values 0-1).', use_container_width=True) # clamp=True is default
        st.write(f"Shape after normalization: {normalized_img_array.shape}, Min/Max pixel: {normalized_img_array.min():.2f}/{normalized_img_array.max():.2f}, Dtype: {normalized_img_array.dtype}")

        # 4. Reshape for the model (1, 28, 28, 1)
        img_array_reshaped = np.expand_dims(normalized_img_array, axis=0)
        img_array_reshaped = np.expand_dims(img_array_reshaped, axis=-1)
        st.write(f"Final shape for model: {img_array_reshaped.shape}")

        # Make prediction
        with st.spinner('Predicting...'):
            prediction = model.predict(img_array_reshaped)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            st.success(f"Predicted Digit: {digit}")
            st.write(f"Confidence: {confidence:.2f}")
            st.bar_chart(prediction[0])

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())
else:
    st.info("Please upload an image file.")
