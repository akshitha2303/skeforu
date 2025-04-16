# 🔬 SKEFORU - Scientific Keyphrase Extractor with Visuals

SKEFORU is an interactive web application designed to extract **keyphrases** from scientific documents (PDF, DOC, DOCX) using **TF-IDF** and visualize them through multiple formats such as **word clouds**, **donut charts**, **radial bar charts**, and **histograms**.

Built with **Streamlit**, this project includes secure **user authentication**, **activity logging**, and a clean user interface, making it ideal for researchers, students, and professionals who want quick insights from their research content.

--- 

# 🚀 Features

- 🔐 **Secure Login/Register** system with password hashing and salt encryption
- 📄 Upload support for **PDF**, **DOC**, and **DOCX** files
- 🧠 **TF-IDF-based Keyphrase Extraction** with unigrams to trigrams
- 📊 Multiple **data visualizations**: WordCloud, Donut Chart, Radial Bar, and Histogram
- 📁 **Activity tracking** per user session with history viewing
- 🎯 Customizable number of keyphrases via slider (5 to 50)
- 🧹 Intelligent text preprocessing and token frequency mapping

---

# 🛠️ Technologies Used

- **Frontend & UI:** Streamlit
- **Backend:** Python, SQLite
- **Security:** Cryptography, PBKDF2 HMAC encryption
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **NLP & ML:** Scikit-learn (TF-IDF), Regex

---

# 📦 Requirements

# Install dependencies using `pip install -r requirements.txt`

pip install -r requirements.txt

# Clone the Repository

pip install -r requirements.txt

# Run Streamlit App

streamlit run Fafa1.py

---

# 💡Future Enhancements

- Use advanced keyphrase extraction techniques (e.g., RAKE, YAKE, KeyBERT)

- Add theme customization for charts

- Improve multi-user dashboard for data insights

