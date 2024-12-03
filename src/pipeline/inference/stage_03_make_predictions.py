import tensorflow as tf

def make_predictions(transformed_prices):
    # Load your trained LSTM model
    model =  tf.keras.models.load_model('artifacts/model_trainer/model/best_5day_model.keras')

    # make prediction
    prediction = model.predict(transformed_prices)

    return prediction

