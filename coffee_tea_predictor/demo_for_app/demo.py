import streamlit as st
import joblib
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Import the Naive Bayes model that uses CountVectoriser
model = joblib.load('naive_bayes_cvec.pkl')
# Import CountVectoriser
cvec = joblib.load('cvec.pkl')

# Global variables that do not need to be redeclared
tokenizer_regex = RegexpTokenizer("\s[\w]+")
lemmatizer = WordNetLemmatizer()


def classify_text(text):
    
    # Tokenise and Lemmatize as done for the model training
    text_tokens_initial = tokenizer_regex.tokenize(text)
    text_tokens_revised = [token.strip() for token in text_tokens_initial]    
    text_tokens_revised = [lemmatizer.lemmatize(token) for token in text_tokens_revised]
    
    text_processed = " ".join(text_tokens_revised)

    X_test_cvec = cvec.transform([text_processed])
    result = model.predict(X_test_cvec)[0]

    if result == 0: # 0 indicates tea
        result_phrase = "Are you a Tea Drinker?"
    else:
        result_phrase = "Are you a Coffee Drinker?"
    return result_phrase, result

# Streamlit application layout
st.title('DPRC Tea or Coffee Predictor')

st.write("Enter a text to predict whether it aligns more with a tea drinker or a coffee drinker.")

user_input = st.text_area("Enter Text Here", "")
if st.button('Analyze'):
    if user_input:
        with st.spinner("Please wait while it brews..."):
            time.sleep(2) # Add timed delay to feedback if the prediction is complete
            prediction_text, prediction= classify_text(user_input)

        st.write(f"Prediction: {prediction_text}")
        if prediction == 0:
            st.image('tea.gif')
        else:
            st.image('coffee.gif')
    else:
        st.write("Do you need Coffee or Tea? Please enter some text to analyze.")