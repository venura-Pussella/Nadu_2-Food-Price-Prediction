from flask import Flask, render_template, request
import os
from prediction import run_prediction_pipeline

app = Flask(__name__)  # initializing a Flask app

@app.route('/', methods=['GET'])  # route to display the home page
def home_page():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    try:
        # Call your training script (make sure it is callable)
        os.system("python main.py")
        return "Training Successful!"
    except Exception as e:
        return f"Error during training: {str(e)}"
    
@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Making predictions using the run_prediction_pipeline
            comparison_df = run_prediction_pipeline()

            # Convert the comparison DataFrame columns to lists
            dates = comparison_df['date'].tolist()
            predicted_values = comparison_df['predicted_value'].tolist()
            real_values = comparison_df['real_value'].tolist()

            # Rendering the template with the prediction results (all predictions)
            return render_template('results.html', dates=dates, predicted_values=predicted_values, real_values=real_values)

        except Exception as e:
            return f"Error during prediction: {str(e)}"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)