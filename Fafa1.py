import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import streamlit as st
from docx import Document
import hashlib
import os
import json
from datetime import datetime
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sqlite3
import io
from wordcloud import WordCloud
import numpy as np

# Security and Database Setup
def setup_database():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password_hash TEXT, salt TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS activities
                 (username TEXT, timestamp TEXT, file_name TEXT, keyphrases TEXT, 
                  FOREIGN KEY (username) REFERENCES users(username))''')
    conn.commit()
    return conn

def generate_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_data(data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

def register_user(username: str, password: str, conn):
    salt = os.urandom(16)
    key = generate_key(password, salt)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                 (username, key.decode(), salt.hex()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str, conn):
    c = conn.cursor()
    c.execute("SELECT password_hash, salt FROM users WHERE username=?", (username,))
    result = c.fetchone()
    if result:
        stored_key = result[0].encode()
        salt = bytes.fromhex(result[1])
        key = generate_key(password, salt)
        return key.decode() == stored_key.decode()
    return False

def save_activity(username: str, file_name: str, keyphrases_df: pd.DataFrame, conn):
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    keyphrases_json = keyphrases_df.to_json()
    c.execute("INSERT INTO activities (username, timestamp, file_name, keyphrases) VALUES (?, ?, ?, ?)",
             (username, timestamp, file_name, keyphrases_json))
    conn.commit()

def get_user_activities(username: str, conn):
    c = conn.cursor()
    c.execute("SELECT timestamp, file_name, keyphrases FROM activities WHERE username=? ORDER BY timestamp DESC",
             (username,))
    return c.fetchall()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Function to compute TF-IDF scores
def compute_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))  # Unigrams to trigrams
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    return pd.DataFrame({'KeyPhrase': feature_names, 'Importance': tfidf_scores}).sort_values(by='Importance', ascending=False), vectorizer

# Streamlit app
def main():
    st.title("Scientific Keyphrase Extractor")
    
    # Initialize database connection
    conn = setup_database()
    
    # Session state for user authentication
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # Login/Register Section
    if not st.session_state.username:
        st.subheader("Login/Register")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            if st.button("Login"):
                if verify_user(login_username, login_password, conn):
                    st.session_state.username = login_username
                    st.success("Login successful!")
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            reg_username = st.text_input("New Username")
            reg_password = st.text_input("New Password", type="password")
            reg_confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                if reg_password != reg_confirm_password:
                    st.error("Passwords do not match!")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long!")
                else:
                    if register_user(reg_username, reg_password, conn):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists!")
    else:
        st.write(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.username = None
            st.rerun()
        
        # Main Application
        st.write("Extract keyphrases from scientific documents with stunning visuals!")
        
        # File Upload Section
        uploaded_file = st.file_uploader("Drag/Drop/Upload Your Scientific Content (PDF/Doc/Docx)", type=["pdf","docx","doc"])
        
        # Keyphrase Selection Slider
        num_keyphrases = st.slider("Select Number of Keyphrases to Extract", min_value=5, max_value=50, value=20, step=5)
        
        # Extract Keyphrases Button
        if st.button("Extract Keyphrases"):
            if uploaded_file:
                # Extract Text from Uploaded File
                with st.spinner("Extracting text and processing..."):
                    file_type = uploaded_file.name.split('.')[-1].lower()  # Determine file type

                    if file_type == "pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    elif file_type in ["doc", "docx"]:  # Added support for DOC/DOCX
                        text = extract_text_from_docx(uploaded_file)
                    else:
                        st.error("Unsupported file format!")  # Added error handling
                        return
                    
                    # Proceed with text preprocessing and keyphrase extraction
                    cleaned_text = preprocess_text(text)
                    tfidf_df, vectorizer = compute_tfidf(cleaned_text)

                    # Correct Frequency Calculation using the vectorizer's tokenization
                    tokenizer = vectorizer.build_analyzer()
                    tokens = tokenizer(cleaned_text)
                    word_counts = pd.Series(tokens).value_counts()

                    # Map token counts to the DataFrame
                    tfidf_df['Frequency'] = tfidf_df['KeyPhrase'].map(word_counts).fillna(0).astype(int)

                    # Normalize Importance to Percentage
                    tfidf_df['Importance (%)'] = (tfidf_df['Importance'] / tfidf_df['Importance'].max()) * 100
                    tfidf_df.drop(columns=['Importance'], inplace=True)  # Drop the raw importance scores

                    # Top N Keyphrases
                    tfidf_top = tfidf_df[['KeyPhrase', 'Frequency', 'Importance (%)']].head(num_keyphrases)
                
                st.success("Keyphrase Extraction Successful!")
                
                # Display Keyphrases in Textual Form
                st.subheader("Keyphrases")
                st.dataframe(tfidf_top)

                # Save activity
                save_activity(st.session_state.username, uploaded_file.name, tfidf_top, conn)

                #Visualize as Word Cloud
                st.subheader("Keyphrase Word Cloud")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                tfidf_top.set_index('KeyPhrase')['Importance (%)'].to_dict()
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

                # Visualize as Donut Pie Chart
                st.subheader("Keyphrase Importance (Donut Chart)")
                fig1, ax1 = plt.subplots()
                ax1.pie(tfidf_top['Importance (%)'], labels=tfidf_top['KeyPhrase'], autopct='%1.1f%%', startangle=140)
                ax1.axis('equal')
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig = plt.gcf()
                fig.gca().add_artist(centre_circle)
                st.pyplot(fig1)

                #Visualize as Radial Bar Chart
                st.subheader("Radial Keyphrase Bar Chart")

                # Create Radial Data
                angles = np.linspace(0, 2 * np.pi, len(tfidf_top), endpoint=False)
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
                bars = ax.bar(angles, tfidf_top['Frequency'], color=plt.cm.viridis(tfidf_top['Importance (%)'] / 100), alpha=0.7)

                # Add Keyphrase Labels
                ax.set_xticks(angles)
                ax.set_xticklabels(tfidf_top['KeyPhrase'], fontsize=8, rotation=45)
                st.pyplot(fig)

                # Visualize as Histogram
                st.subheader("Keyphrase Frequency (Histogram)")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                sns.barplot(x=tfidf_top['KeyPhrase'], y=tfidf_top['Frequency'], palette='viridis', ax=ax2)
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
                ax2.set_title("Keyphrase Frequency")
                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("KeyPhrase")
                st.pyplot(fig2)

            else:
                st.error("Please upload a valid file!")

        # Display Past Activities
        st.subheader("Your Past Activities")
        activities = get_user_activities(st.session_state.username, conn)
        
        if activities:
            for timestamp, file_name, keyphrases_json in activities:
                with st.expander(f"{file_name} - {timestamp}"):
                    keyphrases_df = pd.read_json(keyphrases_json)
                    st.dataframe(keyphrases_df)
                    
                    # Display all visualizations for past activity
                    # Word Cloud
                    st.subheader("Keyphrase Word Cloud")
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                        keyphrases_df.set_index('KeyPhrase')['Importance (%)'].to_dict()
                    )
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)

                    # Donut Chart
                    st.subheader("Keyphrase Importance (Donut Chart)")
                    fig1, ax1 = plt.subplots()
                    ax1.pie(keyphrases_df['Importance (%)'], labels=keyphrases_df['KeyPhrase'], autopct='%1.1f%%', startangle=140)
                    ax1.axis('equal')
                    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                    fig = plt.gcf()
                    fig.gca().add_artist(centre_circle)
                    st.pyplot(fig1)

                    # Radial Bar Chart
                    st.subheader("Radial Keyphrase Bar Chart")
                    angles = np.linspace(0, 2 * np.pi, len(keyphrases_df), endpoint=False)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
                    bars = ax.bar(angles, keyphrases_df['Frequency'], 
                                color=plt.cm.viridis(keyphrases_df['Importance (%)'] / 100), 
                                alpha=0.7)
                    ax.set_xticks(angles)
                    ax.set_xticklabels(keyphrases_df['KeyPhrase'], fontsize=8, rotation=45)
                    st.pyplot(fig)

                    # Histogram
                    st.subheader("Keyphrase Frequency (Histogram)")
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    sns.barplot(x=keyphrases_df['KeyPhrase'], 
                              y=keyphrases_df['Frequency'], 
                              palette='viridis', 
                              ax=ax2)
                    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
                    ax2.set_title("Keyphrase Frequency")
                    ax2.set_ylabel("Frequency")
                    ax2.set_xlabel("KeyPhrase")
                    st.pyplot(fig2)
        else:
            st.info("No past activities found.")

if __name__ == "__main__":
    main()