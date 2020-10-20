
from keras.models import load_model
from keras import backend as K

import pickle
from tensorflow.keras.preprocessing import text, sequence
model_folder = "./models/"
model_name = "01_BiLSTM_Glove"


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

# model path
model_file_path = "./models/01_BiLSTM_Glove_best.h5"

def get_model():
    global model
    model = load_model(model_file_path, 
                       custom_objects={'f1_m': f1_m,           
                                       'precision_m': precision_m,
                                       'recall_m' : recall_m})
    print("Model Loaded")
    return model


def process_new_text(new_text):
    # from tensorflow.keras.preprocessing import text, sequence
    # tokenizer = text.Tokenizer(num_words = max_features, lower = True)
    tokenizer = None
    with open(f"{model_folder}{model_name}_tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)

    new_text_tokenized = tokenizer.texts_to_sequences([new_text])
    new_text_tokenized_paded = sequence.pad_sequences(new_text_tokenized, maxlen=max_text_length)
    return new_text_tokenized_paded

print("\nLoading classifier model ...")
get_model()

user_input = "Trump sucks. He must die!"
print("Processing input text")
user_input_processed = process_new_text(user_input)
print(f"\nProcessed text \n {user_input_processed}")

print("\nGetting prediction")
prediction = model.predict([user_input_processed]).tolist()

print(f"prediction: {prediction}")

print(f"\nModel: {model}")

# print(f"\nModel weights: {model.get_weights()}")

