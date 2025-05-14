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
        return None

model = load_model()

st.title("Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if model is None:
    st.stop()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                # Preprocess the image to fit the model's input requirements
                # 1. Resize to 28x28 pixels
                img_resized = image.resize((28, 28))

                # 2. Convert image to numpy array
                img_array = np.array(img_resized)

                # 3. Invert colors (MNIST is white digit on black background)
                #    Pillow images from upload might be black on white.
                #    This step might need adjustment based on your input image characteristics.
                #    If your drawing canvas naturally produces white on black, you might skip inversion.
                # img_array = 255 - img_array # Uncomment if inversion is needed

                # 4. Normalize the image (scale pixel values to 0-1)
                img_array = img_array.astype("float32") / 255.0

                # 5. Reshape for the model (1, 28, 28, 1)
                #    (1 sample, 28x28 pixels, 1 channel for grayscale)
                img_array_reshaped = np.expand_dims(img_array, axis=0) # Add batch dimension
                img_array_reshaped = np.expand_dims(img_array_reshaped, axis=-1) # Add channel dimension


                # Make prediction
                prediction = model.predict(img_array_reshaped)
                digit = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success(f"Predicted Digit: {digit}")
                st.write(f"Confidence: {confidence:.2f}")
                st.bar_chart(prediction[0])

    except Exception as e:
        st.error(f"An error occurred during processing or prediction: {e}")
else:
    st.info("Please upload an image file.")
