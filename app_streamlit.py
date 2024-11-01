import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from PIL import Image
from prediction import run_prediction_pipeline
from predict_7_days import predict_future_prices  # Import your prediction pipeline

# Set the title of the Streamlit app
st.title("🌾 Pettah Market Nadu 2 Rice Price Prediction System")

# Tabs for different sections of the dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔄 Re-train Model", "📈 Day-by-Day Prediction", "📅 7-Day Prediction", "📊 Model Metrics", "🖼️ Insights"])

# Tab 1: Re-train Model
with tab1:
    st.header("🔄 Re-train Model")
    
    # The model will only re-train if this button is clicked
    if st.button("Start Re-training"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Training in progress...")

        try:
            progress_bar.progress(10)
            status_text.text("Loading data...")
            time.sleep(2)
            progress_bar.progress(30)
            status_text.text("Building model...")
            time.sleep(2)
            progress_bar.progress(50)
            status_text.text("Training model...")
            time.sleep(2)
            progress_bar.progress(80)
            status_text.text("Finalizing...")
            os.system("python main.py")  # Assuming you trigger the pipeline here
            progress_bar.progress(100)
            st.success("Training Successful!")
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Tab 2: Day-by-Day Prediction
with tab2:
    st.header("📈 Day-by-Day Prediction - Model 1")
    
    # Trigger day-by-day prediction only when button is clicked
    if st.button("Run Day-by-Day Prediction"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Prediction in progress...")

        try:
            progress_bar.progress(10)
            comparison_df = run_prediction_pipeline()  # Get prediction data
            progress_bar.progress(70)

            # Display DataFrame
            st.subheader("Prediction Results (Real vs. Predicted)")
            st.write(comparison_df)

            # Plot Real vs. Predicted Values
            st.subheader("Prediction vs. Real Values")
            fig, ax = plt.subplots()
            ax.plot(comparison_df['date'], comparison_df['real_value'], label='Real Value', color='teal')
            ax.plot(comparison_df['date'], comparison_df['predicted_value'], label='Predicted Value', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Real vs Predicted Prices')
            ax.legend()
            plt.xticks(rotation=90)
            st.pyplot(fig)

            progress_bar.progress(100)
            st.success("Prediction Complete!")
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Tab 3: 7-Day Prediction
with tab3:
    st.header("📅 7-Day Price Prediction - Model 1")
    
    # Trigger 7-day prediction only when button is clicked
    if st.button("Run 7-Day Prediction"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Prediction for Next 7 days in progress...")

        try:
            progress_bar.progress(10)
            predicted_df = predict_future_prices()  # Get 7-day prediction data
            progress_bar.progress(70)

            # Display DataFrame
            st.subheader("Prediction Results for the Next 7 Days")
            st.write(predicted_df)

            # Plot Predicted Prices for the next 7 days
            st.subheader("7-Day Price Prediction")
            fig, ax = plt.subplots()
            ax.plot(predicted_df['date'], predicted_df['predicted_value'], label='Predicted Price', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Predicted Price')
            ax.set_title('7-Day Price Prediction')
            ax.legend()
            plt.xticks(rotation=90)
            st.pyplot(fig)

            progress_bar.progress(100)
            st.success("Prediction Complete!")
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Tab 4: Model Metrics
with tab4:
    st.header("📊 Model Evaluation Metrics - Model 1")
    
    # Trigger model metrics loading only when button is clicked
    if st.button("Show Model Metrics"):
        try:
            # Load the metrics file
            metrics_path = os.path.join("artifacts", "model_evaluation", "metrics.xlsx")
            metrics_df = pd.read_excel(metrics_path)

            # Display the metrics DataFrame
            st.subheader("Model Evaluation Metrics Data")
            st.write(metrics_df)
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")

# Tab 5: Insights (Images)
with tab5:
    st.header("🖼️ Future Input Features to the Model")

    try:
        # List of image files for insights
        image_files = [
            "Weekly_Monthly Features Creation.png",
            "Seasonal_Feature.png",
            "Current_Model_LongTerm Prediction_Output.png"   
        ]
        # Dropdown to select an image
        selected_image = st.selectbox(
            "📁 Select an insight image to display:",
            options=image_files,
            index=0
        )

        if selected_image:
            # Load the selected image
            image_path = os.path.join("data", "insights", selected_image)
            image = Image.open(image_path)

            # Display the image with a caption
            st.image(
                image, 
                caption=f"📝 Insights: {selected_image.replace('_', ' ').replace('.png', '')}", 
                use_column_width=True
            )

    except FileNotFoundError:
        st.error("Image not found. Please check the file path.")
    except Exception as e:
        st.error(f"Error showing Insights: {str(e)}")
