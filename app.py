from flask import Flask, render_template, request, url_for
from sklearn import preprocessing
import joblib
import numpy as np

app = Flask(__name__)

def prediction(lst, model):
    if(model == "Random Forest"):
        rf = joblib.load("model/modelDT.joblib")
        pred_value = rf.predict(lst)
    elif(model == "Decision Tree"):
        rf = joblib.load("model/modelDT.joblib")
        pred_value = rf.predict(lst)
    elif(model == "Logistic Regression"):
        rf = joblib.load("model/modelLR.joblib")
        pred_value = rf.predict(lst)
    elif(model == "K Nearest"):
        rf = joblib.load("model/modelKN.joblib")
        pred_value = rf.predict(lst)

    return pred_value


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')


@app.route('/form', methods=['GET'])
def classification():
    return render_template('form.html')

@app.route('/randomForest', methods=['GET'])
def rf():
    return render_template('rf.html')

@app.route('/decisionTree', methods=['GET'])
def dt():
    return render_template('dt.html')

@app.route('/logisticRegression', methods=['GET'])
def lr():
    return render_template('lr.html')

@app.route('/kNerest', methods=['GET'])
def knn():
    return render_template('knn.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    pred_value = 0
    if request.method == 'POST':
        gender = request.form['gender']
        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        age = request.form['age']
        drivinglicense = request.form['drivinglicense']
        if drivinglicense == 'Yes':
            drivinglicense = 1
        else:
            drivinglicense = 0

        vehicleage = request.form['vehicleage']
        if vehicleage == '1-2 Year':
            vehicleage = 0
        elif vehicleage == '1 Year':
            vehicleage = 1
        else:
            vehicleage = 2

        vehicledamage = request.form['vehicledamage']
        if vehicledamage == 'Yes':
            vehicledamage = 1
        else:
            vehicledamage = 0

        vintage = request.form['vintage']

        annualpremium = request.form['annualpremium']

        model = request.form['model_selection']

        feature_list = []

        feature_list.append(gender)
        feature_list.append(int(age))
        feature_list.append(int(drivinglicense))
        feature_list.append(vehicleage)
        feature_list.append(vehicledamage)
        feature_list.append(int(vintage))
        feature_list.append(float(annualpremium))

        feature_list = np.array(feature_list).reshape((1,7))
        # feature_list = [[1, 44, 1,	2,	1,	40454.0,	217]]

        pred_value = prediction(feature_list, model)
        # pred_value = model.predict([[1, 44, 1, 0, 1, 500, 0]])  
        # result_str = str(pred_value[0])

    return render_template('result.html', pred_value=str(pred_value[0]))


if __name__ == '__main__':
    app.run(debug=True)
