# SensorShield: Leveraging AI-Driven Cybersecurity to Safeguard Sensor Networks in Electric Vehicles

### Overview  
CodeVolt SensorShield is a Streamlit-based web application designed for **cybersecurity in electric vehicles (EVs)** using AI-driven anomaly detection techniques. The website provides attack simulations, One class-SVM and LSTM based real-time anomaly detection in CAN (Control area Network), LSTM-based battery spoofing detection, and live anomaly visualization to enhance EV security. 

### Features  
- **Secure Login System** for EV users.
- **Synthetic Attack Generator** for CAN Bus security testing.
- **Real-time anomaly detection** using **One-Class SVM**.
- **Deep Learning-based anomaly detection** using a pre trained **LSTM Autoencoder**(`lstm_autoencoder.h5`).
- **Live Graph Visualization** of anomaly scores.
- **Battery Spoofing Detection** using a pre-traine LSTM model(`battery.h5`).

### Preview 
https://drive.google.com/file/d/1rVNMi8UYEs9T_Mle0LMCWjRgiMvod9mQ/view?usp=sharing

### Project Structure  
```bash
CodeVolt/
│── 📄 lstm_autoencoder.h5          # Pretrained LSTM model for anomoly detection
│── 📄 battery.h5          # Pretrained LSTM model for detecting attacks on battery
│── 📄 CAN.csv             # Input dataset            
│── 📄 app1.py            
│── 📄 app2.py            # Main script for running the app
│── 📄 requirements.txt    # Dependencies
│── 📄 scalar.pkl          
│── 📄 README.md           # Project documentation
│── 📄 License
```
---
###  Installation & Setup  
1. **Clone the repository**  
```sh
git clone https://github.com/RakshaMiglani/CodeVolt.git  
cd CodeVolt
```
2. **Create a virtual environment (recommended)**  
```sh
python -m venv env  
source env/bin/activate  # On Windows: env\Scripts\activate
```
3. **Install dependencies**  
```sh
pip install -r requirements.txt
```
4. **Run the anomaly detection script**  
```sh
streamlit run app2.py
```
## **Customization**
### **Modify Database Credentials**
Update the following function in `app2.py`:
```python
def get_db_connection():
    return mysql.connector.connect(
        host="your_host",
        user="your_username",
        password="your_password",
        database="your_database"
    )
```

### **Change Dataset**
Replace `CAN.csv` with your own dataset.

### **Modify LSTM Model**
Retrain and replace:
- `lstm_autoencoder.h5`
- `battery.h5`

Use:
```python
model.save("lstm_autoencoder.h5")
pickle.dump(scaler, open("scaler.pkl", "wb"))
```
---

## **Usage Guide**
### **Login Page**
1. Enter your **Vehicle Number** and **Password**.
2. Click **Login**.
3. If credentials are valid, you will be redirected to the **dashboard**.

## **Dashboard Features**
### **1. Attack Generation**
- Select an **Attack Type**: `fuzzy`, `spoofing`, `replay`, `DoS`.
- Enter the **number of attack samples**.
- Click **Generate Attack Data**.
- Download the attack data as a CSV.

### **2️. Real-time Anomaly Detection**
- **Input sensor values manually** or **simulate live data**.
- Uses **One-Class SVM** for anomaly detection.
- Displays **Anomaly Score** and detection status.

### **3️. LSTM-based Anomaly Detection**
- Upload a pre-trained LSTM model (`lstm_autoencoder.h5`).
- The model analyzes sensor data for **anomalous patterns**.
- Generates **reconstruction error distribution** and flags anomalies.

### **4️ Battery Spoofing Detection**
- Uses an **LSTM-based model** (`battery.h5`) to detect voltage anomalies.
- Generates **attack data** and detects inconsistencies in **EV battery voltage**.

### **5️. Real-Time Anomaly Visualization**
- Displays **live graphs** of anomaly scores.
- Continuously updates anomaly detection results in real time.


## **Technical Details**
### **Machine Learning Models**
1. **One-Class SVM (OC-SVM)**:
   - Detects **anomalous CAN bus sensor readings**.
   - Trained on normal sensor data to classify outliers.
2. **LSTM Autoencoder**:
   - Learns **normal vehicle behavior** and reconstructs expected values.
   - High reconstruction error indicates **anomalous activity**.
3. **Battery Spoofing Detector**:
   - Uses **LSTM to analyze voltage patterns**.
   - Detects spoofed or tampered battery readings.

### **Dataset**
- The app uses **[CAN.csv]([https://www.google.com](https://www.kaggle.com/datasets/ankitrajsh/can-bus-anomaly-detection-dataset))** as the dataset.
- It contains sensor readings such as **accelerometer values, current, pressure, voltage, and temperature**.

### **Security Considerations**
- **Passwords are stored securely in MySQL.**
- **Session state is used** to prevent unauthorized dashboard access.
- **Anomaly scores and attack data are computed dynamically**, ensuring robustness.

---

### Dependencies  
- Python 3.8+
- MySQL (for login authentication)
- Streamlit
- TensorFlow
- Scikit-learn
- Pandas, NumPy, Matplotlib
- Requests (for fetching online resources)
- Pickle (for saving and loading machine learning models)

### License  
MIT License  
