from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('decision_tree_model_final.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    # Get input values from the form
    daily_sleep_hours = float(request.form['1'])  # Convert to float
    weekly_self_study_hours = float(request.form['2'])  # Convert to float
    
    # Prepare the input array for the model
    arr = np.array([[daily_sleep_hours, weekly_self_study_hours]])
    
    # Make prediction
    pred = model.predict(arr)
    
    # Render the result in after.html
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)
