import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------- Page Config & Styling ---------------------
# --------------------- Page Config & Styling ---------------------
st.set_page_config(page_title="🏠 Bengaluru House Price Predictor", page_icon="🏠")

def set_background(jpg_file_path):
    with open(jpg_file_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        padding: 2rem;
        border-radius: 1rem;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.2);
        color: white;
    }}
    label, h1, h2, h3, h4, h5, h6, div, p {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("background1.jpg")  # Ensure the image path is correct

# --------------------- Database Functions ---------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT
                )""")
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result and verify_password(password, result[0])

init_db()

# --------------------- Session State ---------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

def switch_page(new_page):
    st.session_state.page = new_page

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model_pipeline = load_model()

# --------------------- Load Dataset ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df.dropna(inplace=True)
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else np.nan)

    def convert_sqft_to_num(x):
        try:
            if isinstance(x, str):
                if '-' in x:
                    tokens = x.split('-')
                    return (float(tokens[0]) + float(tokens[1])) / 2
                match = re.findall(r"\d+\.?\d*", x)
                if match:
                    return float(match[0])
            return float(x)
        except:
            return np.nan

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df.dropna(subset=['total_sqft', 'bhk'], inplace=True)
    df = df[(df['total_sqft'] / df['bhk']) >= 250]
    df = df[df['bath'] < df['bhk'] + 3]
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    df = df[df['price_per_sqft'] < df['price_per_sqft'].quantile(0.98)]

    df['total_rooms'] = df['bhk'] + df['bath'] + df['balcony']
    df['sqft_per_bhk'] = df['total_sqft'] / df['bhk']

    location_counts = df['location'].value_counts()
    rare_locations = location_counts[location_counts < 10].index
    df['location'] = df['location'].apply(lambda x: 'Other' if x in rare_locations else x)

    return df

df = load_data()

# --------------------- Login/Register/Forgot UI ---------------------
if not st.session_state.logged_in:

    if st.session_state.page == "login":
        st.title("🏠 Bengaluru House Price Predictor")
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.success(f"Welcome, {username}!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid Username or Password")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("👤 New user?")
            if st.button("Create Account"):
                switch_page("register")

        with col2:
            st.markdown("🔑 Forgot password?")
            if st.button("Forgot Password?"):
                switch_page("forgot")

    elif st.session_state.page == "register":
        st.title("📝 Register Account")
        username = st.text_input("Create Username")
        password = st.text_input("Create Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if password == confirm:
                try:
                    conn = sqlite3.connect("users.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                              (username, hash_password(password)))
                    conn.commit()
                    conn.close()
                    st.success("Registered successfully! You can now login.")
                    switch_page("login")
                except sqlite3.IntegrityError:
                    st.error("Username already exists.")
            else:
                st.warning("Passwords do not match.")

        if st.button("Back to Login"):
            switch_page("login")

    elif st.session_state.page == "forgot":
        st.title("🔒 Reset Password")

        username = st.text_input("Enter your username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.button("Reset Password"):
            if new_password != confirm_password:
                st.warning("Passwords do not match.")
            else:
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username = ?", (username,))
                if c.fetchone():
                    c.execute("UPDATE users SET password = ? WHERE username = ?",
                              (hash_password(new_password), username))
                    conn.commit()
                    conn.close()
                    st.success("✅ Password reset successful. You can now login.")
                    switch_page("login")
                else:
                    st.error("❌ Username not found.")

        if st.button("Back to Login"):
            switch_page("login")

# --------------------- Prediction UI ---------------------
else:
    st.title("🏠 Bengaluru House Price Predictor")
    st.markdown("---")
    st.subheader("🏡 Predict House Price")

    sqft = st.number_input("Total Square Feet", value=1000.0)
    bath = st.number_input("Bathrooms", value=2)
    balcony = st.number_input("Balconies", value=1)
    bhk = st.number_input("Bedrooms (BHK)", value=3)
    location = st.selectbox("Location", sorted(df['location'].unique()))

    if st.button("Predict Price"):
        input_df = pd.DataFrame([{
            'total_sqft': sqft,
            'bath': bath,
            'balcony': balcony,
            'bhk': bhk,
            'location': location,
            'total_rooms': bhk + bath + balcony,
            'sqft_per_bhk': sqft / bhk
        }])

        price_log = model_pipeline.predict(input_df)[0]
        price = np.expm1(price_log)

        st.markdown(
            f"""
            <h3 style='color:#ffffff; background-color:#333333; padding: 15px; border-radius: 8px; text-align: center;'>
                💰 Estimated House Price: ₹{price:.2f} Lakhs
            </h3>
            <p style='text-align: center; margin-top: 10px;'>
                <a href='https://www.google.com/maps/search/{location.replace(' ', '+')}+Bengaluru'
                   target='_blank'
                   style='color:#00BFFF; font-weight:bold; text-decoration: none;'>
                   📍 View {location} on Google Maps
                </a>
            </p>
            """,
            unsafe_allow_html=True
        )

        similar_properties = df[
            (df['location'] == location) &
            (df['bhk'] == bhk) &
            (df['bath'] == bath) &
            (df['balcony'] == balcony)
        ]

        if not similar_properties.empty and 'availability' in similar_properties.columns:
            availability_values = similar_properties['availability'].dropna().unique()
            availability_text = ", ".join(availability_values[:3])

            if any("ready" in val.lower() for val in availability_values):
                status_message = "✅ This configuration is commonly <b>Ready to Move</b>."
            else:
                status_message = f"📅 Similar properties show availability like: <b>{availability_text}</b>"
        else:
            status_message = "❌ No similar match found in current listings based on your input."

        st.markdown(
            f"""
            <div style='background-color:#222; color:#eee; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;'>
                {status_message}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("🎯 Actual vs Predicted Prices")

    X = df[['total_sqft', 'bath', 'balcony', 'bhk', 'location', 'total_rooms', 'sqft_per_bhk']]
    y = np.log1p(df['price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_actual = np.expm1(y_test)
    y_pred_log = model_pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    score = r2_score(y_test_actual, y_pred)

    st.markdown(f"**Model Accuracy (R² Score):** `{score * 100:.2f}%`")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test_actual, y=y_pred, ax=ax, color='purple', s=25, alpha=0.7)
    ax.plot([y_test_actual.min(), y_test_actual.max()],
            [y_test_actual.min(), y_test_actual.max()],
            'k--', lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (Lakhs)")
    ax.set_ylabel("Predicted Price (Lakhs)")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("You have been logged out.")
        st.rerun()
