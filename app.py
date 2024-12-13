import streamlit as st
import os
import pandas as pd
import altair as alt
from inference import inference
from real_price import results

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
    predictions_df= None
    try:
        predictions_df , input_df = inference()
        st.success("Inference successful!")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
    return predictions_df , input_df

def show_real_prices():
    try:
        results_df = results()
        st.success("Show Real Prices successful!")
    except Exception as e:
        st.error(f"Error during show real prices: {str(e)}")
    return results_df

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Model Evaluation","Re-Train - Full Train", "Predict Prices","Real Prices"])

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
        print("Running Prediction...")
        predictions_df, input_df = get_predictions()

        if predictions_df is not None and input_df is not None:

            # Ensure correct data types for input_df
            input_df['date'] = pd.to_datetime(input_df['date'], errors='coerce')
            input_df['pettah_average'] = pd.to_numeric(input_df['pettah_average'], errors='coerce')
            input_df['Stage'] = 'Past'

            # Add 'Stage' column for chart grouping
            predictions_df['Stage'] = 'Predicted'

            # Rename columns for consistent display
            predictions_df = predictions_df.rename(columns={'predicted_value': 'pettah_average'})

            # Combine past and predicted prices
            combined_df = pd.concat([input_df, predictions_df], ignore_index=True)

            # Drop rows with missing values
            combined_df.dropna(subset=['date', 'pettah_average'], inplace=True)

            # Display chart and table
            st.subheader("Prediction Results")

            # Render the Altair line chart
            col = 'pettah_average'
            line_chart = create_line_chart(combined_df, col)
            st.altair_chart(line_chart, use_container_width=True)

            # Display the input (past prices) table
            st.write("Past Prices Table")
            st.dataframe(input_df)

            # Display the predictions table
            st.write("Prediction Table")
            st.dataframe(predictions_df)
        else:
            st.warning("No predictions or past prices available. Please try again.")

if options == "Real Prices":
    st.title("View Real vs Predicted Prices")
    if st.button("Show Real Prices"):
        results_df = show_real_prices()
        if results_df is not None:
            results_df['date'] = pd.to_datetime(results_df['date'], errors='coerce')
            results_df['predicted_value'] = pd.to_numeric(results_df['predicted_value'], errors='coerce')
            results_df['real_value'] = pd.to_numeric(results_df['real_value'], errors='coerce')

            # Add a 'Stage' column to differentiate between real and predicted
            results_df['Stage'] = 'Real'  # This will apply to all rows initially

            # Create a combined DataFrame
            combined_df = results_df.copy()
            combined_df['price'] = combined_df['real_value']
            predicted_part = combined_df[['date', 'predicted_value']].copy()
            predicted_part.rename(columns={'predicted_value': 'price'}, inplace=True)
            predicted_part['Stage'] = 'Predicted'
            
            # Combine the real and predicted prices into one DataFrame
            final_df = pd.concat([combined_df, predicted_part], ignore_index=True)
            final_df.sort_values('date', inplace=True)  # Ensure the dates are sorted

            # Display chart
            col = 'price'
            line_chart = create_line_chart(final_df, col)
            st.altair_chart(line_chart, use_container_width=True)

            # Display a single table with both real and predicted prices
            st.subheader("Real and Predicted Prices Table")
            st.dataframe(final_df[['date', 'price', 'Stage']])
        else:
            st.warning("Failed to load prices. Please check the data source.")
