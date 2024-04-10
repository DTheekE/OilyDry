import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def preprocess_image(image):
    image = load_img(image, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0
    return image_array

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('skin_type_detection_model.h5')
    return model

# Streamlit app
def main():
    st.title("Skin Type Detection")
    st.write("Upload an image to detect the skin type")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = preprocess_image(uploaded_file)

        # Load the model
        model = load_model()

        # Make predictions
        prediction = model.predict(image)

        # Display the prediction and confidence
        confidence = tf.reduce_max(prediction).numpy()
        if confidence > 0.50:
            predicted_class = "Oily"
        else:
            predicted_class = "Dry"

        st.image(uploaded_file, caption=f"Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()
