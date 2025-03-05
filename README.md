### SensorShield: Leveraging AI-Driven Cybersecurity to Safeguard Sensor Networks in Electric Vehicles

### Overview  
CodeVolt is a **real-time voltage anomaly detection system** using an **LSTM-based autoencoder**. It detects anomalies in CAN (Control area network) and provides **attack simulations** for different cyber threats, including **fuzzy, spoofing, replay, and DoS attacks**.  

### Features  
- **Pretrained LSTM Model** (`lstm_autoencoder.h5`) for anomaly detection  
- **Pretrained LSTM voltage analysis model** (`battery.h5`) for detecting attacks on battery  
- **Attack generation module** for cybersecurity threat simulations  
- **Interactive data visualization** with Matplotlib  animations

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
python app2.py
```

### How It Works  
1. Loads **data** from `CAN.csv`  
2. **Preprocesses** the data using `scalar.pkl`  
3. Loads the pretrained **LSTM model (`lstm_autoencoder.h5`)**  
4. Detects anomalies based on **reconstruction error**  
5. Generates **attack simulations** and identifies voltage anomalies
6. Loads the pretrained **LSTM model (`battery.h5`)**  
7. Detects attacks on battery based on **time series voltage data**  
8. Visualises the anomolus data using animated graphs from mathplotlib

### Attack Simulations  
Supports:  
- **Fuzzy Attack**  
- **Spoofing Attack**  
- **Replay Attack**  
- **DoS Attack**

### Dependencies  
- Python 3.8+  
- TensorFlow  
- NumPy, Pandas  
- Matplotlib  

### License  
MIT License  
