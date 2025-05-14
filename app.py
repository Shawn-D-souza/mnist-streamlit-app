import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from streamlit_cropper import st_cropper # Import the cropper component

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

st.title("Handwritten Digit Recognizer with Manual Crop")
st.write("Upload an image, crop the digit using the interactive box, and get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Stop execution if the model failed to load
if model is None:
    st.stop()

# Initialize cropped_image variable
cropped_image = None

if uploaded_file is not None:
    try:
        # Open the uploaded image
        img = Image.open(uploaded_file)

        st.write("Adjust the box below to crop the digit:")

        # --- Add the st_cropper component ---
        # Replace the initial st.image display with the cropper
        cropped_image = st_cropper(
            img,
            realtime_update=True, # Show updates as you drag the box
            box_color='#00F0FF', # Color of the cropping box
            aspect_ratio=(1,1), # Encourage user to select a square region
            return_type='image', # Return a PIL Image object
            key='digit_cropper'  # Unique key for the component
        )
        # --- End of cropper component ---

        # Display the cropped image preview returned by the component
        # This will update as the user interacts with the cropper
        if cropped_image: # Check if the cropper has returned an image
            st.write("Preview of the selected crop:")
            st.image(cropped_image, use_container_width=True) # Display the cropped image

            # Add the Predict button here, visible only after an image is cropped
            if st.button('Predict Cropped Digit'):
                with st.spinner('Predicting...'):
                    # --- Preprocessing Steps for the CROPPED Image ---

                    # 1. Convert the cropped image to grayscale (if it's not already)
                    #    st_cropper usually returns RGB, so this is necessary.
                    cropped_image_gray = cropped_image.convert('L')

                    # 2. Resize the grayscale cropped image to 28x28 pixels
                    #    The model requires this specific input size.
                    img_resized = cropped_image_gray.resize((28, 28))

                    # 3. Convert image to numpy array
                    img_array = np.array(img_resized)

                    # 4. Invert colors if needed (typical for black digit on white background crops)
                    #    MNIST model expects white digits on a black background.
                    #    If your cropped digit is black on white, UNCOMMENT this:
                    img_array = 255 - img_array # <--- UNCOMMENTED for typical black-on-white crops

                    # --- Optional Debugging: Display the 28x28 image fed to the model ---
                    # Convert the numpy array back to a PIL Image (scaling back to 0-255)
                    # img_for_display = Image.fromarray((img_array * 255).astype(np.uint8))
                    # st.subheader("Image fed to the model (28x28):")
                    # Use a small fixed width for display as it's tiny
                    # st.image(img_for_display, width=100, caption="Processed Image (should be white digit on black)")
                    # --------------------------------------------------------------------


                    # 5. Normalize the image (scale pixel values to 0-1)
                    img_array = img_array.astype("float32") / 255.0

                    # 6. Reshape for the model (1, 28, 28, 1)
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
    st.info("Please upload an image file to begin.")
