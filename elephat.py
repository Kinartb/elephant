import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load a pre-trained model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def identify_animal(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    return decoded_predictions

def generate_chart(predictions):
    # Extract top categories and their probabilities
    categories = [pred[1] for pred in predictions]
    probabilities = [pred[2] for pred in predictions]
    
    # Create a bar chart
    fig, ax = plt.subplots()
    ax.barh(categories, probabilities, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Top Predicted Animals/Birds')
    
    return fig

def main():
    st.title("Animal/Bird Identification App")

    # Sidebar
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an option", ["Upload Image", "View Data Visualization"])

    if option == "Upload Image":
        st.header("Upload an Image")

        # Image upload widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                # Display progress and status
                with st.spinner("Identifying the animal..."):
                    predictions = identify_animal(image)
                    animal_name = predictions[0][1]  # Get the name of the top prediction
                    st.write(f"Identified Animal/Bird: {animal_name}")
                     # Display the chart with top predictions
                    fig = generate_chart(predictions)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")
                
    elif option == "View Data Visualization":
        st.header("Data Visualization")
    
        st.write("Upload an image to view data visualization related to the predictions.")

if __name__ == "__main__":
    main()