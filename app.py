import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model, scaler, and label encoder
model_path = 'random_forest_model.pkl'
scaler_path = 'scaler.pkl'
label_encoder_path = 'label_encoder.pkl'

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

try:
    model = load_pickle(model_path)
    scaler = load_pickle(scaler_path)
    label_encoder = load_pickle(label_encoder_path)
    st.success("Model, scaler, and label encoder loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or files: {e}")

# Streamlit app
st.title("Travel Destination Predictor")

# User inputs
num_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=1)
num_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
destination_type = st.selectbox("Destination Type", ["Beach", "Nature", "City", "Adventure"])  # Update with actual types

# Prepare the input for prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'NumberOfAdults': [num_adults],
        'NumberOfChildren': [num_children],
        'Type': [destination_type]
    })

    # Encode the destination type
    try:
        if destination_type not in label_encoder.classes_:
            st.error(f"Error in encoding: '{destination_type}' is not a recognized destination type.")
            st.stop()
        
        input_data['TypeEncoded'] = label_encoder.transform(input_data['Type'])
    except ValueError as e:
        st.error(f"Error in encoding: {e}")
        st.stop()

    # Prepare data for scaling (ensure all necessary features are included)
    input_data_prepared = pd.DataFrame({
        'NumberOfAdults': [num_adults],
        'NumberOfChildren': [num_children],
        'DestinationTypeEncoded': [input_data['TypeEncoded'][0]]  # Ensure correct feature name
    })

    # Scale the data
    try:
        input_data_scaled = scaler.transform(input_data_prepared)
    except ValueError as e:
        st.error(f"Error in scaling data: {e}")
        st.stop()
    
    # Make prediction
    try:
        prediction = model.predict(input_data_scaled)
        st.write(f"Predicted Destination: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
