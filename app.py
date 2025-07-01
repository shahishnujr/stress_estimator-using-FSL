import matplotlib
matplotlib.use('Agg')  

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CSV_FILE = 'users.csv'
SENSOR_LOG = 'manual_sensor_log.csv'

class ProtoNet(nn.Module):
    def __init__(self, feature_dim=1024):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096, track_running_stats=False),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, track_running_stats=False),
            nn.Dropout(0.3),
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim, track_running_stats=False)
        )

    def forward(self, x):
        if self.training or x.shape[0] > 1:
            return self.encoder(x)
        else:
            for layer in self.encoder:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                x = layer(x)
            return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ProtoNet().to(device)
model.load_state_dict(torch.load("best_protonet_model.pth", map_location=device))
model.eval()

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
scaler = joblib.load("scaler.pkl")


def get_latest_sensor_values():
    try:
        if os.path.exists(SENSOR_LOG):
            with open(SENSOR_LOG, newline='') as csvfile:
                rows = list(csv.DictReader(csvfile))
                for row in reversed(rows):
                    try:
                        sys_bp = float(row['sys_bp'])
                        dia_bp = float(row['dia_bp'])
                        hr = float(row['heart_rate'])
                        spo2 = float(row['spo2'])
                        return sys_bp, dia_bp, hr, spo2
                    except ValueError:
                        continue
    except Exception as e:
        print("❌ Failed to load latest sensor record:", e)
    return '', '', '', ''

def predict_stress(sys_bp, dia_bp, heart_rate, spo2):
    input_data = np.array([[sys_bp, dia_bp, heart_rate, spo2]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    def euclidean_distance(a, b):
        return torch.cdist(a, b, p=2)

    with torch.no_grad():
        embedding = model(input_tensor)
        class_prototypes = torch.stack([
            model(torch.tensor(X_train[y_train == i], dtype=torch.float32).to(device)).mean(0)
            for i in np.unique(y_train)
        ])
        dists = euclidean_distance(embedding, class_prototypes)
        pred_label = torch.argmin(dists, dim=1).item()

    if sys_bp > 140 and  dia_bp>90 and heart_rate > 100:
        pred_label = 1  

    label_map = {0: "Normal", 1: "Stress High"}
    return label_map[pred_label]


def generate_prediction_graph():
    timestamps = []
    systolic = []
    diastolic = []
    heart_rates = []
    spo2s = []

    try:
        with open(SENSOR_LOG, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            if not rows:
                return None
                
            for row in rows[-10:]:  # Last 10 readings
                timestamps.append(row['timestamp'].split()[1])
                systolic.append(float(row['sys_bp']))
                diastolic.append(float(row['dia_bp']))
                heart_rates.append(float(row['heart_rate']))
                spo2s.append(float(row['spo2']))

        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, systolic, label='Systolic BP')
        plt.plot(timestamps, diastolic, label='Diastolic BP')
        plt.plot(timestamps, heart_rates, label='Heart Rate')
        plt.plot(timestamps, spo2s, label='SpO₂')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    except Exception as e:
        print("❌ Graph generation error:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with open(CSV_FILE, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['email'] == email and row['password'] == password:
                    session['user'] = row['name']
                    flash('Login successful!', 'success')
                    return redirect(url_for('predict'))
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        user_exists = False
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['email'] == email:
                        user_exists = True
                        break
        if user_exists:
            flash('Email already registered.', 'warning')
        else:
            with open(CSV_FILE, 'a', newline='') as csvfile:
                fieldnames = ['name', 'email', 'password']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if os.stat(CSV_FILE).st_size == 0:
                    writer.writeheader()
                writer.writerow({'name': name, 'email': email, 'password': password})
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash('Please log in to access prediction.', 'warning')
        return redirect(url_for('login'))

    result = None
    sys_bp = dia_bp = heart_rate = spo2 = ''

    if request.method == 'POST':
        if 'fetch' in request.form:
            sys_bp, dia_bp, heart_rate, spo2 = get_latest_sensor_values()
        else:
            try:
                form_sys = request.form.get('sys_bp', '')
                form_dia = request.form.get('dia_bp', '')
                form_hr = request.form.get('heart_rate', '')
                form_spo2 = request.form.get('spo2', '')
                
                if not all([form_sys, form_dia, form_hr, form_spo2]):
                    flash('Please fill in all fields with valid numbers.', 'danger')
                else:
                    sys_bp = float(form_sys)
                    dia_bp = float(form_dia)
                    heart_rate = float(form_hr)
                    spo2 = float(form_spo2)
                    result = predict_stress(sys_bp, dia_bp, heart_rate, spo2)
            except ValueError:
                flash('Please enter valid numeric values for all fields.', 'danger')

    graph_url = generate_prediction_graph()

    return render_template('predict.html', result=result,
                           sys_bp=sys_bp, dia_bp=dia_bp,
                           heart_rate=heart_rate, spo2=spo2,
                           graph_url=graph_url)

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
