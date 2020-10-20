import numpy as np
import pandas as pd 
import tensorflow as tf
from flask import Flask, render_template, redirect, url_for, request, Response, jsonify
from keras.models import load_model

import pickle
from tensorflow.keras.preprocessing import text, sequence
model_folder = "./models/"
# model_name = "01_BiLSTM_Glove"
multilabel_model_name = "01_BiLSTM_Glove_MULTILABEL"

binary_model_name = "01_BiLSTM_Glove_BINARY"

tokenizer = None


app = Flask(__name__)


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Constants
max_features = 48201      # no. of worlds to embed, rest will be ignored
max_text_length = 400       # max. length of comments, shorter comments will be padded 
embed_size = 300            # work embedding size


def get_binary_model():
    global binary_model
    binary_model = load_model(f"{model_folder}{binary_model_name}_best.h5", 
                        custom_objects={'f1_m': f1_m,           
                                        'precision_m': precision_m,
                                        'recall_m' : recall_m})
    print("Model Loaded")
    return binary_model


print("Loading Binary classifier model ...")
get_binary_model()

def get_multilabel_model():
    global multilabel_model
    multilabel_model = load_model(f"{model_folder}{multilabel_model_name}_best.h5", 
                        custom_objects={'f1_m': f1_m,           
                                        'precision_m': precision_m,
                                        'recall_m' : recall_m})
    print("Model Loaded")
    return multilabel_model

print("Loading Mulitlabel classifier model ...")
get_multilabel_model()


def binary_process_new_text(new_text):
    # from tensorflow.keras.preprocessing import text, sequence
    # tokenizer = text.Tokenizer(num_words = max_features, lower = True)
    tokenizer = None
    with open(f"{model_folder}{binary_model_name}_tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    new_text_tokenized = tokenizer.texts_to_sequences([new_text])
    new_text_tokenized_paded = sequence.pad_sequences(new_text_tokenized, maxlen=max_text_length)
    return new_text_tokenized_paded

def multilabel_process_new_text(new_text):
    # from tensorflow.keras.preprocessing import text, sequence
    # tokenizer = text.Tokenizer(num_words = max_features, lower = True)
    tokenizer = None
    with open(f"{model_folder}{multilabel_model_name}_tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    new_text_tokenized = tokenizer.texts_to_sequences([new_text])
    new_text_tokenized_paded = sequence.pad_sequences(new_text_tokenized, maxlen=max_text_length)
    return new_text_tokenized_paded

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/toxicity_classifier", methods=["GET"])
def toxicity_classifier():
    return render_template("toxicity_classifier.html")


@app.route("/predict", methods=["POST"])
def predict():
    # get comment text
    user_input = request.form['comment_text']
    print(f"user_input: {user_input}")
    data = {
        "text_to_predict" : user_input
    }

    # Binary
    # process the text to be predicted
    binary_new_text_processed = binary_process_new_text(data["text_to_predict"])
    print(binary_new_text_processed)

    binary_prediction = binary_model.predict(binary_new_text_processed)

    print(binary_prediction)


    feedback_text = ""

    if round(binary_prediction[0][0]) == 0:  # Toxic
        # MultiLabel Prediction

        # process the text to be predicted
        multilabel_new_text_processed = multilabel_process_new_text(data["text_to_predict"])
        print(multilabel_new_text_processed)

        # make the prediction
        multilabel_prediction = multilabel_model.predict(multilabel_new_text_processed).tolist()
        
        print(f"prediction: {multilabel_prediction}")

        
        predictions = {
                'toxic' : multilabel_prediction[0][0],
                'severe_toxic' : multilabel_prediction[0][1],
                'obscene' : multilabel_prediction[0][2],
                'threat' : multilabel_prediction[0][3],
                'insult' : multilabel_prediction[0][4],
                'identity_hate' : multilabel_prediction[0][5]
            }


        #  Chart stuff --------------
        prediction_values = [multilabel_prediction[0][0], multilabel_prediction[0][1], multilabel_prediction[0][2], multilabel_prediction[0][3], multilabel_prediction[0][4], multilabel_prediction[0][5]]
        prediction_label = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']


        chartID =  'chartID'
        chart_type = 'bar'
        chart_height = 300
        fill_color = 'red'

        chart = {"renderTo": chartID, "type": chart_type, "height": chart_height, "fill" : fill_color}
    
        series = [{"name": 'Probability', "data": prediction_values}]


        print(f"series: {series}")
        print(f"prediction_values: {prediction_values}")

        title = {"text": 'How likely does your text contain the following toxic characteristics?'}
        xAxis = {"categories": prediction_label }
        yAxis = {"title": {"text": 'Probability'}, "min":0.0, "max":1.0}

        # Chart stuff ends
        
        feedback_text = f"Text is {round(prediction_values[0]*100,2)}% likely to be toxic. Please rephrase your text."

        return render_template("toxicity_classifier.html", 
                                test_data = multilabel_prediction,   # new stuff added below
                                chartID = chartID, 
                                chart = chart, 
                                series = series, 
                                title = title, 
                                xAxis = xAxis, 
                                yAxis = yAxis,
                                user_input = user_input,
                                feedback_text = feedback_text)
    else:
        # result = "Text is neutral."
        #  Chart stuff --------------
        prediction_values = [0,0,0,0,0,0]
        prediction_label = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

        chartID =  'chartID'
        chart_type = 'bar'
        chart_height = 300
        fill_color = 'red'

        chart = {"renderTo": chartID, "type": chart_type, "height": chart_height, "fill" : fill_color}

    
        series = [{"name": 'Probability', "data": prediction_values}]


        print(f"series: {series}")
        print(f"prediction_values: {prediction_values}")

        title = {"text": 'How likely does your text contain the following toxic characteristics?'}
        xAxis = {"categories": prediction_label }
        yAxis = {"title": {"text": 'Probability'}, "min":0.0, "max":1.0}


        feedback_text = "Thank you for being civil."



        return render_template("toxicity_classifier.html",
                                # test_data = multilabel_prediction,   # new stuff added below
                                chartID = chartID, 
                                chart = chart, 
                                series = series, 
                                title = title, 
                                xAxis = xAxis, 
                                yAxis = yAxis,
                                user_input = user_input,
                                feedback_text = feedback_text)


@app.route("/perspective", methods=["GET"])
def perspective():
    return render_template("perspective.html")


if __name__ == "__main__":
    app.run(debug = False)

    

