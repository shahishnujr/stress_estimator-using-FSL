# Stress Predictor using few shot learning

This project is an **IoT-enabled health monitoring system** that predicts user stress levels based on real-time physiological sensor data (blood pressure, heart rate, SpO₂). The system uses a custom-trained Prototypical Network (ProtoNet) deep learning model and provides a user-friendly web interface for monitoring and prediction.

## 🚀 Features

- **IoT Integration:** Collects live data from health sensors (via serial port).
- **Machine Learning:** Uses a ProtoNet deep learning model for stress prediction.
- **Web Dashboard:** User authentication, live data display, and prediction visualization (Flask).
- **Data Logging:** Stores all sensor readings for analysis and visualization.
- **Manual & Automatic Input:** Supports both live sensor fetching and manual data entry.
- **Recent Trends:** Visualizes last 10 readings with Matplotlib.

## 📊 Model Performance

- **Test Accuracy:** 96.95%
- **Precision:** 0.97
- **Recall:** 0.96
- **F1 Score:** 0.96

The ProtoNet model was trained and evaluated on the physiological dataset, achieving high accuracy and strong precision/recall for stress detection.

## 📂 Directory Structure

```
IOT_Final-1.pynb # Colab file using for the few shot model training and implementation with accuracy of 96.95%
STRESS_PREDICTOR/
├── .venv/ # Python virtual environment
├── sketch_apr20a/ # Arduino/Embedded code for sensors
├── static/css/style.css # CSS styling
├── templates/ # Flask HTML templates
│ ├── base.html
│ ├── index.html
│ ├── login.html
│ ├── predict.html
│ └── register.html
├── app.py # Main Flask application
├── serdata.py # Serial data logger (sensor to CSV)
├── best_protonet_model.pth # Trained ProtoNet model weights
├── scaler.pkl # Scaler for feature normalization
├── X_train.npy # Training data (features)
├── y_train.npy # Training data (labels)
├── manual_sensor_log.csv # Main sensor data log (used by app)
├── users.csv # User registration/login data
```
## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stress_predictor.git
cd stress_predictor
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv

Windows:
.venv\Scripts\activate

Linux/Mac:
source .venv/bin/activate

```

### 3. Install Dependencies
```bash
pip install flask torch numpy pandas scikit-learn matplotlib joblib tqdm sentence-transformers setfit datasets
```

### 4. Connect and Run Sensor Logger

- Connect your sensor hardware (Arduino, etc.) to the specified COM port.
- Run the serial logger to collect data:
```bash
python serdata.py
```

This will continuously log sensor readings to `manual_sensor_log.csv`.

### 5. Start the Web Application
```bash
python app.py
```

- Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🖥️ Usage

- **Register** a new user or **login** with existing credentials.
- **Fetch Latest** to auto-fill the latest sensor readings.
- **Predict** to get the stress level using the AI model.
- **View Trends**: See recent sensor data visualized as a graph.
- **Logout** when finished.

## ⚙️ Model & Data

- **ProtoNet Model:**  
  - Trained on physiological data (`X_train.npy`, `y_train.npy`)  
  - Model weights: `best_protonet_model.pth`
  - Scaler: `scaler.pkl`
- **Manual Data Logging:**  
  - All readings are stored in `manual_sensor_log.csv` for audit and visualization.

## 🧑‍💻 Project Structure

| File/Folder             | Purpose                                      |
|-------------------------|----------------------------------------------|
| `app.py`                | Flask web server & ML inference              |
| `serdata.py`            | Serial port sensor data logger               |
| `best_protonet_model.pth` | Trained AI model weights                   |
| `scaler.pkl`            | Data normalization scaler                    |
| `X_train.npy`/`y_train.npy` | Model training data                     |
| `manual_sensor_log.csv` | Main sensor readings log                     |
| `users.csv`             | User authentication data                     |
| `templates/`            | HTML templates for Flask                     |
| `static/css/`           | CSS styling                                  |
| `sketch_apr20a.ino`     | (Optional) Arduino/MCU sensor code           |




