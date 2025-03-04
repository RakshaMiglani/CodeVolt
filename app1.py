import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import random
import time
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed # type: ignore
import matplotlib.pyplot as plt

# Function to get database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="$Sri123456",
        database="your_database"
    )

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'login'  # Initial page

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-image: url('https://wallpaperaccess.com/full/3295839.jpg');
            background-size: cover;
            background-position: center;
            color: white;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #00ffcc;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #ffffff;
        }
        .login-box {
            width: 40%;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.6);
        }
        .stTextInput > div > div > input {
            background-color: white;
            color: black;
        }
        .stButton button {
            background-color: #00ffcc;
            color: black;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Conditional Page Display based on Session State
if st.session_state.page == 'login':

    # UI Layout
    st.markdown('<p class="title">🔒 SensorShield: AI-Powered Cybersecurity for EVs</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Protecting Sensor Networks in Electric Vehicles Using AI-Driven Anomaly Detection</p>', unsafe_allow_html=True)

    st.markdown("---")  # Horizontal line for separation
    st.image("https://www.tesla.com/sites/default/files/modelsx-new/social/model-s-hero-social.jpg", width=600)

    # Login Box
    with st.container():
        st.subheader("🔑 Secure Vehicle Login")

        vehicle_number = st.text_input("🚗 Enter Vehicle Number", key="vehicle_number")
        password = st.text_input("🔐 Enter Password", type="password", key="password")

        if st.button("Login"):
            if not vehicle_number or not password:
                st.error("⚠️ Please enter both Vehicle Number and Password.")
            else:
                try:
                    # Check database for matching credentials
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM users WHERE vehicle_number = %s AND password = %s", (vehicle_number, password))
                    user = cursor.fetchone()
                    cursor.close()
                    conn.close()

                    if user:
                        st.success("✅ Login successful! Redirecting...")
                        st.session_state.page = 'dashboard'  # Update session state
                        st.rerun()  # Force re-run to display dashboard
                    else:
                        st.error("❌ Invalid credentials. Please try again.")

                except Exception as e:
                    st.error("⚠️ Server error! Please try again later.")
                    st.text(f"Error: {str(e)}")  # Log error details
        col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="🚗 EVs on Road", value="10M+", delta="Rising 🚀")
    with col2:
        st.metric(label="⚠️ Cyber Threats", value="5K+", delta="Increasing 📈")
    with col3:
        st.metric(label="🔒 AI Security Success", value="99.9%", delta="Improved 💡")

    st.markdown("---")  # Horizontal line
    st.write("💡 **Future of AI in EV Security:** SensorShield aims to create a **secure ecosystem** where AI-driven monitoring enhances safety and resilience against cyber threats in electric vehicles.")

    # Footer
    st.markdown('<p style="text-align:center; color:lightgray;">&copy; 2025 SensorShield | AI-Powered EV Cybersecurity</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close login box

elif st.session_state.page == 'dashboard':
    df = pd.read_csv("CAN.csv")  # Change this to your actual dataset filename

features=[
        "datetime", "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
        "Temperature", "Thermocouple", "Volume Flow RateRMS","Voltage"]

# Attack generation functions
def generate_attack(attack_type, num_samples=10):
    attack_data = []

    if attack_type == "fuzzy":
        for _ in range(num_samples):
            attack_data.append({
                "tag": f"Fuzzy_{random.randint(1000, 9999)}",
                "datetime": pd.Timestamp.now(),
                "Accelerometer1RMS": random.uniform(0, 50),
                "Accelerometer2RMS": random.uniform(0, 50),
                "Current": random.uniform(0, 10),
                "Pressure": random.uniform(10, 100),
                "Temperature": random.uniform(20, 100),
                "Thermocouple": random.uniform(10, 50),
                "Voltage": random.uniform(200, 250),
                "Volume Flow RateRMS": random.uniform(5, 50),
                "Attack": "Fuzzy"
            })

    elif attack_type == "spoofing":
        sampled_rows = df.sample(n=num_samples)
        for _, row in sampled_rows.iterrows():
            attack_data.append({
                "tag": row["tag"],
                "datetime": pd.Timestamp.now(),
                "Accelerometer1RMS": row["Accelerometer1RMS"] + random.uniform(-2, 2),
                "Accelerometer2RMS": row["Accelerometer2RMS"] + random.uniform(-2, 2),
                "Current": row["Current"] + random.uniform(-1, 1),
                "Pressure": row["Pressure"] + random.uniform(-5, 5),
                "Temperature": row["Temperature"] + random.uniform(-3, 3),
                "Thermocouple": row["Thermocouple"] + random.uniform(-2, 2),
                "Voltage": row["Voltage"] + random.uniform(-5, 5),
                "Volume Flow RateRMS": row["Volume Flow RateRMS"] + random.uniform(-3, 3),
                "Attack": "Spoofing"
            })

    elif attack_type == "replay":
        sampled_rows = df.sample(n=num_samples)
        for _, row in sampled_rows.iterrows():
            attack_data.append({
                "tag": row["tag"],
                "datetime": pd.Timestamp.now() + pd.Timedelta(seconds=random.randint(10, 300)),
                "Accelerometer1RMS": row["Accelerometer1RMS"],
                "Accelerometer2RMS": row["Accelerometer2RMS"],
                "Current": row["Current"],
                "Pressure": row["Pressure"],
                "Temperature": row["Temperature"],
                "Thermocouple": row["Thermocouple"],
                "Voltage": row["Voltage"],
                "Volume Flow RateRMS": row["Volume Flow RateRMS"],
                "Attack": "Replay"
            })

    elif attack_type == "dos":
        sampled_rows = df.sample(n=num_samples)
        for _ in range(5):  # Simulating flooding
            for _, row in sampled_rows.iterrows():
                attack_data.append({
                    "tag": row["tag"],
                    "datetime": pd.Timestamp.now(),
                    "Accelerometer1RMS": row["Accelerometer1RMS"],
                    "Accelerometer2RMS": row["Accelerometer2RMS"],
                    "Current": row["Current"],
                    "Pressure": row["Pressure"],
                    "Temperature": row["Temperature"],
                    "Thermocouple": row["Thermocouple"],
                    "Voltage": row["Voltage"],
                    "Volume Flow RateRMS": row["Volume Flow RateRMS"],
                    "Attack": "DoS"
                })

    return attack_data

features=[
        "datetime", "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
        "Temperature", "Thermocouple", "Volume Flow RateRMS","Voltage"]
# Convert timestamp to numerical format (if not already)
df["datetime"] = pd.to_datetime(df["datetime"]).astype(int) / 10**9  # Convert to seconds

scaler = StandardScaler()
X_train = df[features]
X_train_scaled = scaler.fit_transform(X_train)

oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
oc_svm.fit(X_train_scaled)

def detect_anomaly(sensor_values):
    sensor_values_scaled = scaler.transform([sensor_values])
    anomaly_score = oc_svm.decision_function(sensor_values_scaled)[0]
    prediction = 1 if anomaly_score >= 60 else -1
    return prediction, anomaly_score

def train_lstm_model(df):
    df["datetime"] = pd.to_datetime(df["datetime"]).astype(int) / 10**9
    data = df[features].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    SEQ_LEN = 10
    X = []
    for i in range(len(data_scaled) - SEQ_LEN):
        X.append(data_scaled[i: i + SEQ_LEN])
    X = np.array(X)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(SEQ_LEN, X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="relu", return_sequences=False),
        RepeatVector(SEQ_LEN),
        LSTM(32, activation="relu", return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(X.shape[2]))
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, X_train, epochs=2, batch_size=32, validation_data=(X_test, X_test), verbose=0)
    X_test_pred = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.abs(X_test_pred - X_test), axis=(1, 2))
    THRESHOLD = np.percentile(reconstruction_error, 95)
    return model, scaler, THRESHOLD

def color_code_anomalies(df):
    colors = {}
    for feature in ["datetime","Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple", "Volume Flow RateRMS", "Voltage"]:
        values = df[feature]
        threshold = values.mean() + values.std()
        
        if values.max() > threshold * 1.2:  # 🔴 Red: Detected anomaly
            colors[feature] = "red"
        elif values.max() > threshold * 1.05:  # 🟡 Yellow: Potential issue
            colors[feature] = "yellow"
        else:  # 🟢 Green: Normal data
            colors[feature] = "green"

    return colors

# Sidebar for main navigation
main_page = st.sidebar.selectbox("Choose a page", ["Attack Generation", "Real Time Anomaly Detection", "Battery Spoofing Detection"])
page = main_page
if main_page == "Attack Generation":
    st.title("🚀 CAN Bus Synthetic Attack Generator 🚀")

    attack_type = st.selectbox("Select Attack Type", ["fuzzy", "spoofing", "replay", "dos"])
    num_samples = st.number_input("Enter Number of Samples to Generate", min_value=1, value=10)

    if st.button("Generate Attack Data"):
        attack_data = generate_attack(attack_type, num_samples)
        attack_df = pd.DataFrame(attack_data)

        st.write("### Generated Attack Data:")
        st.dataframe(attack_df)

        csv = attack_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='attack_data.csv',
            mime='text/csv',
        )

elif main_page == "Real Time Anomaly Detection":
    st.title("🔍 CAN Bus Anomaly Detection 🔍")
    
    # Subpage selection for Anomaly Detection
    subpage = st.radio("Select Anomaly Detection Mode", ["Input Values", "Real-time Simulation"])
    
    if subpage == "Input Values":
        st.subheader("Input Sensor Values")
        
        # Input fields for sensor values
        datetime = st.number_input("Datetime", value=0.0)
        acc1 = st.number_input("Accelerometer1RMS", value=0.0)
        acc2 = st.number_input("Accelerometer2RMS", value=0.0)
        current = st.number_input("Current", value=0.0)
        pressure = st.number_input("Pressure", value=0.0)
        temperature = st.number_input("Temperature", value=0.0)
        thermocouple = st.number_input("Thermocouple", value=0.0)
        volume_flow = st.number_input("Volume Flow RateRMS", value=0.0)
        voltage = st.number_input("Voltage", value=0.0)
        
        if st.button("Detect Anomaly"):
            sensor_values = [datetime, acc1, acc2, current, pressure, temperature, thermocouple, volume_flow, voltage]
            prediction, score = detect_anomaly(sensor_values)
            status = "Anomaly" if prediction == 1 else "Normal"
            st.write(f"Anomaly Score: {score:.3f} | Status: {status}")
    
    elif subpage == "Real-time Simulation":
        st.subheader("Real-time Anomaly Detection Simulation")
        st.write("Simulating real-time anomaly detection...")

        if st.button("Start Anomaly Detection Simulation"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            results_container = st.empty()
            all_results = []

            for i, (_, row) in enumerate(df.sample(n=10).iterrows()):
                values = row[features].tolist()
                prediction, score = detect_anomaly(values)
                status = "Anomaly" if prediction == 1 else "Normal"
                
                # Color coding
                background_color = "#FF4B4B" if prediction == 1 else "#4BFF4B"
                text_color = "white" if prediction == 1 else "black"
                
                # Create the result string
                result = f"""
                <div style="background-color: {background_color}; padding: 10px; border-radius: 5px; color: {text_color}; margin-bottom: 10px;">
                    Anomaly Score: {score:.3f} | Status: {status}<br>
                    {', '.join([f'{col}: {row[col]:.2f}' for col in features])}
                </div>
                """
                
                # Add the result to the list
                all_results.append(result)
                
                # Update the container with all results
                results_container.markdown(''.join(all_results), unsafe_allow_html=True)
                
                progress_bar.progress((i + 1) / 10)
                time.sleep(1)

            st.success("Simulation completed!")

elif main_page == "Battery Spoofing Detection":
    # Anomaly detection setup (keep existing code)
    features = ["datetime","Voltage"]
    st.title("🔍 LSTM-based Voltage Anomaly Detection 🔍")
    
    if 'lstm_model' not in st.session_state:
        with st.spinner("Training LSTM model..."):
            st.session_state.lstm_model, st.session_state.scaler, st.session_state.threshold = train_lstm_model(df)
        st.success("LSTM model trained successfully!")
    
    st.subheader("Generate and Detect Voltage Anomalies")
    
    attack_type = st.selectbox("Select Attack Type", ["fuzzy", "spoofing", "replay", "dos"])
    num_samples = st.number_input("Enter Number of Samples to Generate", min_value=1, value=50, step=1)
    
    if st.button("Generate and Detect Anomalies"):
        attack_data = generate_attack(attack_type, num_samples)
        attack_df = pd.DataFrame(attack_data)
        attack_df["datetime"] = pd.to_datetime(attack_df["datetime"]).astype(int) / 10**9
        
        features_to_scale = ["datetime", "Voltage"]
        numerical_attack_df = attack_df[features_to_scale]
        attack_data_scaled = st.session_state.scaler.transform(numerical_attack_df)
        
        SEQ_LEN = 10
        attack_X = []
        for i in range(len(attack_data_scaled) - SEQ_LEN):
            attack_X.append(attack_data_scaled[i: i + SEQ_LEN])
        attack_X = np.array(attack_X)
        
        attack_pred = st.session_state.lstm_model.predict(attack_X, verbose=0)
        attack_reconstruction_error = np.mean(np.abs(attack_pred - attack_X), axis=(1, 2))
        
        attack_anomalies = attack_reconstruction_error > st.session_state.threshold
        
        st.write(f"Anomalies Detected: {np.sum(attack_anomalies)} out of {len(attack_anomalies)}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(attack_reconstruction_error, label="Reconstruction Error")
        ax.axhline(y=st.session_state.threshold, color='r', linestyle='--', label="Anomaly Threshold")
        ax.set_title(f"Reconstruction Error for {attack_type.capitalize()} Attack")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Reconstruction Error")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Generated Attack Data")
        st.dataframe(attack_df)