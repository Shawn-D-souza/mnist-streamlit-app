import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Function to load the model (cached for performance)
@st.cache_resource
def load_model():
    try:
        # Ensure the model file 'mnist_model.keras' is in the same directory
        # as your Streamlit script, or provide the correct path.
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

        # Display the uploaded image using use_container_width instead of use_column_width
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Add a button to trigger prediction
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                # Preprocess the image to fit the model's input requirements

                # 1. Resize to 28x28 pixels
                img_resized = image.resize((28, 28))

                # 2. Convert image to numpy array
                img_array = np.array(img_resized)

                # 3. Invert colors if necessary (MNIST is white digit on black background)
                #    Uploaded images are often black on white.
                #    Uncomment the line below if your input images are black on white.
                # img_array = 255 - img_array

                # 4. Normalize the image (scale pixel values to 0-1)
                img_array = img_array.astype("float32") / 255.0

                # 5. Reshape for the model (1, 28, 28, 1)
                #    (Batch size 1, 28x28 pixels, 1 channel for grayscale)
                img_array_reshaped = np.expand_dims(img_array, axis=0)    # Add batch dimension
                img_array_reshaped = np.expand_dims(img_array_reshaped, axis=-1) # Add channel dimension


                # Make prediction using the loaded model
                prediction = model.predict(img_array_reshaped)

                # Get the predicted digit (the class with the highest probability)
                digit = np.argmax(prediction)

                # Get the confidence score for the predicted digit
                confidence = np.max(prediction)

                # Display the prediction results
                st.success(f"Predicted Digit: **{digit}**")
                st.write(f"Confidence: **{confidence:.2f}**")

                # Display a bar chart of the prediction probabilities
                st.subheader("Prediction Probabilities:")
                st.bar_chart(prediction[0])

    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")
        st.error("Please ensure the uploaded file is a valid image.")

else:
    # Message shown when no file is uploaded
    st.info("Please upload an image file to get a prediction.")
