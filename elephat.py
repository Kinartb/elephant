import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load a pre-trained model (e.g., MobileNetV2)
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

def preprocess_image(image):
    """Preprocess the uploaded image to match the model input."""
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

def identify_animal(image):
    """Predict the animal in the image using the pre-trained model."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

def generate_chart(predictions):
    """Generate a bar chart of the top predicted categories."""
    categories = [pred[1] for pred in predictions]
    probabilities = [pred[2] for pred in predictions]
    
    fig, ax = plt.subplots()
    ax.barh(categories, probabilities, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Top Predicted Animals/Birds')
    ax.invert_yaxis()  # Invert y-axis to show the highest probability on top
    
    return fig

def main():
    st.title("Sri Lankan Wildlife Identification App")

    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an option", ["Upload Image", "View Data Visualization"])

    if option == "Upload Image":
        st.header("Upload an Image")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with st.spinner("Identifying the animal..."):
                    predictions = identify_animal(image)
                    animal_name = predictions[0][1]
                    st.subheader(f"Identified Animal/Bird: {animal_name}")
                    
                    fig = generate_chart(predictions)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")
                
    elif option == "View Data Visualization":
        st.header("Data Visualization")
        st.write("Upload an image to view data visualization related to the predictions.")

if __name__ == "__main__":
    main()

       
    
      
