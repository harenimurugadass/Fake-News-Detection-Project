import os
import nltk
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from streamlit_option_menu import option_menu

# --- Set NLTK data path (for local use or deployment) ---
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk")
nltk.data.path.append(nltk_data_path)

# --- Optional: Auto download if running locally or on Streamlit Cloud ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load the trained pipeline model ---
pipeline = joblib.load('fake_news_pipeline.pkl')

# --- Preprocessing function ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    clean_text = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(clean_text)

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #0f0f23;
            color: #ffffff;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextArea, .stTextInput {
            background-color: #1e1e2f;
            color: white;
        }
        @keyframes animateHeadline {
            0% { opacity: 0; transform: translateY(10px); }
            50% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-10px); }
        }
        .headline {
            font-size: 60px;
            font-weight: bold;
            color: #7b2cbf;
            font-family: Arial, sans-serif;
            animation: animateHeadline 2s ease-in-out infinite;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Navigation Menu ---
selected_page = option_menu(
    menu_title='',
    options=["Home", "About", "Contact"],
    icons=["house", "info-circle", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "#0f0f23",
            "justify-content": "center"
        },
        "icon": {
            "color": "#ffffff",
            "font-size": "18px"
        },
        "nav-link": {
            "font-size": "18px",
            "color": "#ffffff",
            "padding": "10px",
            "text-align": "center"
        },
        "nav-link-selected": {
            "background-color": "#7b2cbf",
            "color": "#ffffff",
            "border-bottom": "3px solid #ffffff"
        }
    }
)

# --- Home Page ---
if selected_page == "Home":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="headline">🚀 Detect Fake News Instantly</div>', unsafe_allow_html=True)
    st.markdown("""
        <h3 style='text-align: center; color: gray;'>
        Check if a news article is <span style='color:green;'>Real</span> or <span style='color:red;'>Fake</span> in just a few seconds!
        </h3>
    """, unsafe_allow_html=True)

    st.title("📰 Fake News Detection App")
    st.markdown("An ML-powered system using **TF-IDF + LinearSVC** to detect fake vs real news.")

    # Tabs
    tab1, tab2 = st.tabs(["🏠 Home", "📝 Input News"])

    with tab1:
        st.subheader("Welcome to the Fake News Detection App!")
        st.write("""
            This app allows you to check if a news article is real or fake using a trained machine learning model.
            
            - Model: TF-IDF + LinearSVC
            - Dataset: Real & fake news from ISOT + additional sources
            - Preprocessing: Stopword removal, lemmatization
        """)
        st.info("Go to the 'Input News' tab to test a news article!")

    # Input News Tab with prediction result
    with tab2:
        st.subheader("Enter a News Article")
        input_news = st.text_area("Paste the news content here:", height=250)

        if st.button("Submit for Prediction"):
            if input_news.strip() == "":
                st.warning("⚠️ Please enter some news text.")
            else:
                 with st.spinner("Analyzing the news..."):
                      preprocessed = preprocess_text(input_news)
                      prediction = pipeline.predict([preprocessed])[0]

                 if prediction == 1:
                    st.success("✅ This news article is **REAL**.")
                    st.info("The content appears to be factually accurate based on real news patterns.")
                    st.balloons()
                 else:
                    st.error("🚨 This news article is **FAKE**.")
                    st.warning("This content may be misleading or contain misinformation.")
                    st.snow()

# --- About Page ---
elif selected_page == "About":
    st.title("ℹ️ About This App")
    st.write("""
This application is designed to detect whether a news article is REAL or FAKE using Natural Language Processing (NLP) and Machine Learning (ML).

**Technologies Used:**
- TF-IDF Vectorizer
- Linear Support Vector Classifier (LinearSVC)
- Python, scikit-learn, NLTK, Streamlit

**Workflow:**
- Dataset: Real & Fake news from Kaggle + manually added articles
- Preprocessing: Lemmatization, stopwords removal, regex
- Trained on balanced dataset
- Deployed using Streamlit

""")

# --- Contact Page ---
elif selected_page == "Contact":
    st.title("📨 Contact")
    st.write("""
**Developer:** Hareni M  
📧 Email: harenimurugadass@gmail.com  
🔗 LinkedIn: [linkedin.com](https://www.linkedin.com/in/hareni-m-15436721a )  
💻 GitHub: [github.com](https://github.com/harenimurugadass/Fake-News-Detection-Project)
""")

# --- Footer ---
st.markdown(""" 
<hr style="border: 1px solid #7b2cbf;">
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <p style="color: gray; font-size: 16px;">
        © 2025 <b>Fake News Detection App</b> 
    </p>
    <p style="font-size: 14px; color: #aaa;">
        Developed by <a href="https://github.com/harenimurugadass/Fake-News-Detection-Project" style="color: #7b2cbf; text-decoration: none;" target="_blank">Hareni</a>
    </p>
</div>
""", unsafe_allow_html=True)
