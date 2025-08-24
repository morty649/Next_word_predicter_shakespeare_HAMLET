import streamlit as st
import numpy as np
import pandas
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

#Load the LSTM model
model = load_model('next_word_lstm_shakespeare.h5')

#Load the tokenizer
with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)


#Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_length):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_length:
    token_list = token_list[-(max_sequence_length-1):]
  token_list = pad_sequences([token_list],maxlen=max_sequence_length-1,padding='pre')
  predicted = model.predict(token_list,verbose=0)
  predicted_word_index = np.argmax(predicted,axis=1)
  for word,index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None


#streamlit starts here buoy

st.title("Next word prediction with LSTM over shakespeare dataset")
input_text = st.text_input("Enter the sequence of words","to be or not to")
if st.button("predict next word"):
  max_sequence_length = model.input_shape[1]+1
  next_word = predict_next_word(model,tokenizer,input_text,max_sequence_length)
  st.write(f"Next word:{next_word}")

  