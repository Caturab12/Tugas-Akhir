import streamlit as st
import pandas as pd
import pickle
import numpy as np
import streamlit.components.v1 as components
import sidebar
from preprocessing import clean_text, normalized_term, stopwords, stem_text
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer

# Set the page configuration
st.set_page_config(page_title="Analisis Sentimen Website", layout="wide")

# menampilkan sidebar dan mendapatkan pilihan menu
selected = sidebar.show()

if selected == "Dashboard":
    st.image("images/exploration.png", width=85)
    st.markdown("""
    # Sentimen Cerdas 
    Analisis sentimen adalah proses mengumpulkan, memahami, dan mengevaluasi opini, perasaan, atau sikap yang terkandung dalam teks, seperti tweet.
    #####   Tujuan:
    1. Untuk menganalisis sentimen pengguna terhadap kualitas layanan jasa ekspedisi (J&T Express, JNE Express dan Shopee Express) yang diungkapkan melalui media sosial Twitter (X). 
    2. Menggunakan Algoritma Logistic Regression dan LSTM untuk memberikan wawasan yang berharga bagi perusahaan jasa ekspedisi melalui hasil akurasi. 

                
    ##### Manfaat: 
    Pelanggan atau pengguna dapat memperoleh wawasan yang lebih mendalam mengenai tanggapan atau opini masyarakat luas, terutama di media sosial X, terkait kualitas pelayanan yang ditawarkan oleh ketiga jasa ekspedisi (J&T Express, JNE Express dan Shopee Express). Informasi ini memungkinkan pelanggan untuk membuat keputusan yang lebih bijak dalam memilih jasa ekspedisi untuk mengirimkan barang ke tujuan akhir dengan aman. 
    """)

    # Create columns for the indicators
    col1, col2, col3 = st.columns(3)

    # Positive posts indicator
    with col1:
        st.markdown("""
        <div style="border: 1px solid #E0E0E0; padding: 20px; border-radius: 5px; text-align: center;">
            <h3>DATA LATIH</h3>
            <div style="font-size: 50px; color: green;"></div>
            <div style="font-size: 30px;">30%</div>
        </div>
        """, unsafe_allow_html=True)

    # Negative posts indicator
    with col2:
        st.markdown("""
        <div style="border: 1px solid #E0E0E0; padding: 20px; border-radius: 5px; text-align: center;">
            <h3>DATA UJI</h3>
            <div style="font-size: 50px; color: red;"></div>
            <div style="font-size: 30px;">70%</div>
        </div>
        """, unsafe_allow_html=True)

    # Total data indicator
    with col3:
        st.markdown("""
        <div style="border: 1px solid #E0E0E0; padding: 20px; border-radius: 5px; text-align: center;">
            <h3>TOTAL DATA</h3>
            <div style="font-size: 50px; color: blue;">∑</div>
            <div style="font-size: 30px;">2580</div>
        </div>
        """, unsafe_allow_html=True)

if selected == "Klasifikasi":
    # Load model dan vectorizer untuk teks
    with open('tf_idf_feature.pickle', 'rb') as f:
        tf_idf = pickle.load(f)

    model_lstm = load_model('lstm_model.hdf5')

    with open('model_logistic_regression.pickle', 'rb') as f:
        model_lr = pickle.load(f)

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    st.title("Sentiment Analysis 😊😐😕😡")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    input_text = st.text_area('Input Teks Analysis:', placeholder='Masukan Teks')

    if st.button('Analisis'):
        # Preprocessing teks
        clean = clean_text(input_text)
        regexp = RegexpTokenizer(r'\w+|$[0-9]+|S+')
        tokenize = regexp.tokenize(clean)
        normalize = normalized_term(tokenize)
        stopword = stopwords(normalize)
        stemmed = stem_text(stopword)
        final = ' '.join(stemmed)

        # Tampilkan tahapan preprocessing
        st.subheader('Preprocessing Steps:')
        st.write(
            pd.DataFrame({'Step': ['Cleaning', 'Tokenization', 'Normalization', 'Stopword Removal', 'Stemming', 'Final'],
                          'Result': [clean, tokenize, normalize, stopword, stemmed, final]}))

        # Tokenization
        input_sequence = tokenizer.texts_to_sequences([final])
        input_padded = pad_sequences(input_sequence, maxlen=200)

        # LSTM prediction
        prediction_lstm = model_lstm.predict(input_padded)

        # Logistic Regression prediction
        if hasattr(tf_idf, 'transform'):
            input_tf_idf = tf_idf.transform([final])
        else:
            st.error("The tf_idf object does not have the transform method.")

        prediction_lr = model_lr.predict(input_tf_idf)

        sentiment_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        lstm_prediction = sentiment_map[np.argmax(prediction_lstm)]

        st.write('LR :', prediction_lr[0])
        st.write('LSTM :', lstm_prediction)
