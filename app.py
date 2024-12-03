import streamlit as st
import os
import pandas as pd
import altair as alt

# Function to create the line chart
def create_line_chart(metrics_df, col):
    # Format column for y-axis and tooltip
    formatted_col = col.replace('_', ' ').title()
    tooltip_format = ',.2f'  # Format without $ sign

    # Calculate y-axis min and max
    y_min = metrics_df[col].min() - 5
    y_max = metrics_df[col].max() + 5

    # Create Altair line chart
    line_chart = alt.Chart(metrics_df).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y(f'{col}:Q', title=f"{formatted_col} (LKR)", scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color('Stage:N', title='Stage'),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('Stage:N', title='Stage'),
            alt.Tooltip(f'{col}:Q', title=f"{formatted_col} (LKR)", format=tooltip_format),
        ]
    ).interactive()

    return line_chart

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Nadu_2 Price Prediction",
    layout="wide",
)

# Function to trigger training
def run_model_evaluation_training():
    try:
        os.system("python model_evaluation_main.py")  # Adjust the path to your training script if needed
        st.success("model evaluation training successful!")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

def run_model_full_training():
    try:
        os.system("python model_full_training_main.py")  # Adjust the path to your training script if needed
        st.success("Model full training successful!")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

# Function to run predictions
def get_predictions():
    try:
        os.system("python inference.py") 
        st.success("Inference successful!")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

# def show_real_prices():
#     try:
#         os.system("python real_price.py") 
#         st.success("Show Real Prices successful!")
#     except Exception as e:
#         st.error(f"Error during training: {str(e)}")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Model Evaluation","Re-Train - Full Train", "Predict Prices"])

# Home Page
if options == "Home":
    st.title("Welcome to the Food Price Predictions for Next 5 Days")
    # st.write("Use the sidebar to navigate and explore options to train the model or predict future prices.")

# Train Model Page
elif options == "Model Evaluation":
    st.title("Train the Model for Evaluation")
    st.write("Model Performance Evaluation Training Pipeline - Nadu 2")
    if st.button("Start Training - Model Evaluation"):
        run_model_evaluation_training()

# Train Model Page
elif options == "Re-Train - Full Train":
    st.title("Full Train the Model Before Predictions")
    st.write("Trigger Full Training Pipeline - Nadu 2")
    if st.button("Start Training - Full Training"):
        run_model_full_training()

# Predict Future Prices Page
elif options == "Predict Prices":
    st.title("Predict Pettah Market Nadu 2 Prices for Next 5 Days")
    if st.button("Run Prediction"):
        predictions_df = get_predictions()

        if predictions_df is not None:
            # Ensure correct data types
            predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
            predictions_df['predicted_value'] = pd.to_numeric(predictions_df['predicted_value'], errors='coerce')

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
