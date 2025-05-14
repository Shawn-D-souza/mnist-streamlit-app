import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Function to load the model (cached for performance)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mnist_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'mnist_model.keras' is in the correct location.")
        return None

# Load the model when the app starts
model = load_model()

st.title("Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9).")
st.write("For best results, use a clear image of a single digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Stop execution if the model failed to load
if model is None:
    st.stop()

if uploaded_file is not None:
    try:
        # Open the image and convert to grayscale
        image = Image.open(uploaded_file).convert('L')

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Add a button to trigger prediction
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                # --- Preprocessing Steps ---

                # 1. Convert image to grayscale (already done, but good practice)
                #    image = image.convert('L')

                # 2. Pad the image to make it square while preserving aspect ratio
                width, height = image.size
                max_side = max(width, height)

                # Create a new square image with a black background (pixel value 0)
                # MNIST has black background and white digits.
                # If your digits are black on white, you might use white background (255)
                # or handle inversion later. Let's stick to black background for MNIST compatibility.
                padded_image = Image.new('L', (max_side, max_side), 0) # 0 for black background

                # Calculate the position to paste the original image onto the center
                x_offset = (max_side - width) // 2
                y_offset = (max_side - height) // 2

                # Paste the original image onto the center of the padded square image
                padded_image.paste(image, (x_offset, y_offset))

                # 3. Resize the padded square image to 28x28 pixels
                img_resized = padded_image.resize((28, 28))

                # 4. Convert image to numpy array
                img_array = np.array(img_resized)

                # 5. Invert colors if the digits are black on white
                #    MNIST model expects white digits on a black background.
                #    If your drawing results in black digits on a white background, uncomment this:
                # img_array = 255 - img_array

                # 6. Normalize the image (scale pixel values to 0-1)
                img_array = img_array.astype("float32") / 255.0

                # 7. Reshape for the model (1, 28, 28, 1)
                #    (Batch size 1, 28x28 pixels, 1 channel for grayscale)
                img_array_reshaped = np.expand_dims(img_array, axis=0)
                img_array_reshaped = np.expand_dims(img_array_reshaped, axis=-1)

                # --- Prediction ---
                prediction = model.predict(img_array_reshaped)
                digit = np.argmax(prediction)
                confidence = np.max(prediction)

                # --- Display Results ---
                st.success(f"Predicted Digit: **{digit}**")
                st.write(f"Confidence: **{confidence:.2f}**")
                st.subheader("Prediction Probabilities:")
                st.bar_chart(prediction[0])

    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")
        st.error("Please ensure the uploaded file is a valid image.")

else:
    st.info("Please upload an image file to get a prediction.")
