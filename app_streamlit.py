import streamlit as st
import os
import pandas as pd
import altair as alt
from predict import predict_future_prices  # Import your prediction function
import numpy as np

# Function to create the line chart
def create_line_chart(metrics_df, col):
    # Format column for y-axis and tooltip
    formatted_col = col.replace('_', ' ').title()
    tooltip_format = '$,.2f'

    # Calculate y-axis min and max
    y_min = metrics_df[col].min() - 5
    y_max = metrics_df[col].max() + 5

    # Create Altair line chart
    line_chart = alt.Chart(metrics_df).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y(f'{col}:Q', title=f"{formatted_col} ($)", scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color('Stage:N', title='Stage'),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('Stage:N', title='Stage'),
            alt.Tooltip(f'{col}:Q', title=formatted_col, format=tooltip_format),
        ]
    ).interactive()

    return line_chart

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Nadu_2 Price Prediction",
    layout="wide",
)

# Function to trigger training
def run_training():
    try:
        os.system("python main.py")  # Adjust the path to your training script if needed
        st.success("Training successful!")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

# Function to run predictions
def get_predictions():
    try:
        predictions_df = predict_future_prices()  # Get predictions as a DataFrame
        return predictions_df
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Re-Train", "Predict Prices"])

# Home Page
if options == "Home":
    st.title("Welcome to the Food Price Prediction App")
    st.write("Use the sidebar to navigate and explore options to train the model or predict future prices.")

# Train Model Page
elif options == "Re-Train":
    st.title("Train the Model")
    st.write("Click the button below to start the training process.")
    if st.button("Start Training"):
        run_training()

# Predict Future Prices Page
elif options == "Predict Prices":
    st.title("Predict Pettah Market Nadu_2 Rice Prices for Next 5 Days")
    if st.button("Run Prediction"):
        predictions_df = get_predictions()

        if predictions_df is not None:
            # Ensure correct data types
            predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
            predictions_df['predicted_value'] = pd.to_numeric(predictions_df['predicted_value'], errors='coerce')

            # Optional variability for demo purposes
            if st.checkbox("Add random variability to predictions (demo purposes)"):
                predictions_df['predicted_value'] += np.random.uniform(-5, 5, size=len(predictions_df))

            # Add 'Stage' column for chart grouping
            predictions_df['Stage'] = 'Predicted'

            # Drop rows with missing values
            predictions_df.dropna(subset=['date', 'predicted_value'], inplace=True)

            # Display chart and table
            st.subheader("Prediction Results")

            # Render the Altair line chart
            col = 'predicted_value'
            line_chart = create_line_chart(predictions_df, col)
            st.altair_chart(line_chart, use_container_width=True)

            # Display the prediction table
            st.write("Prediction Table")
            st.dataframe(predictions_df)
        else:
            st.warning("No predictions available. Please try again.")
