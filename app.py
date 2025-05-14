import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps # Make sure ImageOps is imported
import numpy as np
import traceback # For detailed error messages

# Function to load the model (cached for performance)
@st.cache_resource
def load_model_cached(): # Renamed to avoid conflict if you had 'load_model' elsewhere
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
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.subheader("1. Original Uploaded Image (Grayscale)")
        st.image(image, caption='Uploaded Grayscale Image.', use_column_width=True)

        # --- Strategy 1: Resizing with Aspect Ratio Preservation and Padding ---
        desired_size = 28
        old_size = image.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img_resized_aspect = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create a new_image with a black background
        final_28x28_img = Image.new("L", (desired_size, desired_size), 0) # 0 for black background
        final_28x28_img.paste(img_resized_aspect, ((desired_size-new_size[0])//2,
                                          (desired_size-new_size[1])//2))
        
        st.subheader("2. Resized to 28x28 (Aspect Preserved, Padded)")
        st.image(final_28x28_img, caption='Processed for Model Input.', use_column_width=True)
        
        # Convert to numpy array
        img_array = np.array(final_28x28_img)
        st.write(f"Shape after padding: {img_array.shape}, Min/Max pixel: {img_array.min()}/{img_array.max()}, Dtype: {img_array.dtype}")

        # 3. Invert colors (if necessary: MNIST is white digit on black background)
        # If after padding, your digit is black and background is white (less likely with black padding),
        # or if the digit itself within the original image was light on a dark background.
        # More commonly, if your original digit was dark on a light background, it will still be dark after padding onto black.
        # So, inversion is usually needed to make the dark digit white.
        st.subheader("3. Inverted Image (Attempting White Digit on Black Background)")
        # This assumes the digit is now darker than the (newly black) background if it was dark originally
        img_inverted = 255 - img_array  # Make dark digits white
        st.image(img_inverted, caption='Inverted Colors.', use_column_width=True)
        st.write(f"Shape after inversion: {img_inverted.shape}, Min/Max pixel: {img_inverted.min()}/{img_inverted.max()}")
        
        processed_img_array = img_inverted # Use the inverted image

        # 4. Normalize the image (scale pixel values to 0-1)
        processed_img_array = processed_img_array.astype("float32") / 255.0
        st.subheader("4. Normalized Image Array (Pixel values 0-1)")
        st.image(processed_img_array, caption='Normalized (Values 0-1).', clamp=True)
        st.write(f"Shape after normalization: {processed_img_array.shape}, Min/Max pixel: {processed_img_array.min()}/{processed_img_array.max()}, Dtype: {processed_img_array.dtype}")

        # 5. Reshape for the model (1, 28, 28, 1)
        img_array_reshaped = np.expand_dims(processed_img_array, axis=0)
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
