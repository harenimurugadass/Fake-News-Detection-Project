# 📰 Fake News Detection Project

A Machine Learning-powered web application that detects whether a given news article is real or fake using Natural Language Processing techniques.

## 📌 About

In the current age of digital media, fake news can spread rapidly and cause significant damage. This project combats misinformation by building an intelligent system that can accurately identify fake news articles using ML and NLP techniques.

This project includes:
- A training pipeline using TF-IDF vectorization and a Linear Support Vector Machine (SVM) classifier.
- A visually styled and user-friendly Streamlit web app for real-time predictions.
- Public deployment for anyone to test.

## ⚙️ How It Works

1. **Preprocessing**: News content is cleaned, stopwords are removed, and lemmatization is applied.
2. **Vectorization**: TF-IDF converts the cleaned text into numerical features.
3. **Prediction**: A pre-trained LinearSVC model classifies the news as `FAKE` or `REAL`.
4. **Result Display**: The result is shown with emojis and animations, with prediction logging.

## 🌟 Features

- 🧠 Fake news detection using ML + NLP
- 📊 Interactive Streamlit web interface
- 🧼 Automatic text preprocessing
- 💬 Animated result messages and emojis
- 📁 Logging of predictions for analysis
- 🌐 Hosted on Streamlit Community Cloud

## 🚀 How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/harenimurugadass/Fake-News-Detection-Project.git
   cd Fake-News-Detection-Project

2. **Install dependencies**
   (Recommended: create a virtual environment)
    ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app**
    ```bash
   streamlit run fake_news_app.py

## 📊 Dataset Info

This project uses the ISOT Fake News Dataset, which includes two CSV files:
- True.csv: Real news articles
- Fake.csv: Fake news articles
- additional_real_news.csv: Supplementary real news content. Some news are fetch from newsapi.org

**Preprocessing steps include:**
- Removing punctuation and digits
- Lowercasing
- Removing stopwords using NLTK
- Lemmatization using WordNetLemmatizer
  
## 🧰 Technologies Used Python

- Python
- Scikit-learn
- NLTK(v3.7)
- Pandas, NumPy, Matplotlib
- Streamlit
- TF-IDF Vectorizer
- LinearSVC (Support Vector Machine)
- Joblib (model serialization)
- HTML/CSS (for UI styling)

## 🌐 Live Demo

Try the app live here: 👉 [Fake News Detector Web App](https://fakenewsdetection-ml-nlp.streamlit.app)

## 📬 Contact

Created by Hareni M
- 🔗 [LinkedIn](https://www.linkedin.com/in/hareni-m-15436721a)
- 📧 harenimurugadass@gmail.com
- 💻 [GitHub Project Link](https://github.com/harenimurugadass/Fake-News-Detection-Project)

Feel free to ⭐️ the repo, fork the project, or open issues. Contributions are welcome!
