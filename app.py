import numpy as np
from flask import Flask, render_template, request
import pickle

model = pickle.load(open('xgb_modeltestfinal99.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_fraud():
    months_as_customer = float(request.form.get('months_as_customer'))
    policy_deductable = float(request.form.get('policy_deductable'))
    umbrella_limit = float(request.form.get('umbrella_limit'))
    capital_gains = float(request.form.get('capital_gains'))
    capital_loss = float(request.form.get('capital_loss'))
    incident_hour_of_the_day = float(request.form.get('incident_hour_of_the_day'))
    number_of_vehicles_involved = float(request.form.get('number_of_vehicles_involved'))
    bodily_injuries = float(request.form.get('bodily_injuries'))
    witnesses = float(request.form.get('witnesses'))
    vehicle_claim = float(request.form.get('vehicle_claim'))
    incident_severity = float(request.form.get('incident_severity'))
    authorities_contacted = float(request.form.get('authorities_contacted'))
    collision_type = float(request.form.get('collision_type'))
    # fraud_reported = float(request.form.get('fraud_reported'))

    # Make prediction
    input_features = np.array([months_as_customer, policy_deductable, capital_loss,
                               incident_hour_of_the_day, number_of_vehicles_involved, capital_gains, witnesses,
                               vehicle_claim, collision_type, incident_severity, authorities_contacted, umbrella_limit,bodily_injuries]).reshape(1, -1)

    result = model.predict(input_features)

    if result == 1:
        result_message = "This is fraudulent."
    else:
        result_message = "This is not fraudulent."

    return render_template('index.html', result=result_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
