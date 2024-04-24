from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
import joblib
import numpy as np
import pickle

app = Flask(__name__)

# This dictionary will act as a temporary storage for user credentials
users = {}

# Define the feature mapping dictionary
fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
pmap = {'icmp': 0, 'tcp': 1, 'udp': 2,'http':3,'ftp':4,'ip':5,'smtp':6}
smap={'http':0,'private':1}

model_paths = {
     'gaussian_naive_bayes': './models/model1.pkl',
    'decision_tree': './models/model2.pkl',
    'random_forest': './models/model3.pkl',
    'svm': './models/model4.pkl',
    'logistic_regression': './models/model5.pkl',
    'gradboost': './models/model6.pkl',
}

models = {name: joblib.load(path) for name, path in model_paths.items()}

def predict_model(model_key, df):
    model = models[model_key]
    return model.predict(df)


@app.route('/')
def index():
    return render_template('signin.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        # Handle sign-in form submission
        # Extract username and password from the form
        username = request.form['username']
        password = request.form['password']
        # Perform sign-in process here (e.g., validate credentials)
        if username in users and users[username] == password:
            return render_template('home.html')  # Redirect to the dashboard page after successful login
        return render_template('signin.html', error='Invalid username or password.')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup form submission
        # Extract username and password from the form
        username = request.form['username']
        password = request.form['password']
        # Perform signup process here (e.g., add user to database)
         # Check if the username already exists
        if username in users:
            return render_template('signup.html', error='Username already exists.')
        # Store the username and password in the dictionary
        users[username] = password
        # After successful signup, redirect to sign-in page
        return render_template('signin.html')
    return render_template('signup.html')  # Render sign-up page


# Define the predict_model function somewhere
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Manually entered data
        # data = [request.form.get(field,type=float) for field in request.form if field != 'model']
        data = [request.form.get(field) for field in request.form if field != 'model']
        # df['flag'] = data['flag'].map(fmap)  # Applying the mapping
        # df['protocol_type'] = data['protocol_type'].map(pmap)
        # print(data)

# Create DataFrame from form data
        # df = pd.DataFrame([data])
        df = pd.DataFrame([data], columns=[field for field in request.form if field != 'model'])
        print(df) 

        # Apply feature mapping
        df['flag'] = df['flag'].map(fmap)
        df['protocol_type'] = df['protocol_type'].map(pmap)
        df['service'] = df['service'].map(smap)
        df.drop(columns='service',inplace=True,axis=1)

        # Print DataFrame for debugging
        # print(df)

        # Predict using selected models
        selected_models = request.form.getlist('model')
        results = {model: predict_model(model, df) for model in selected_models}
        
        
        # Render the results
        return render_template('predict.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
