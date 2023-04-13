import flask
import joblib
import json
import pandas as pd
from data_loder import DataLoader
from models import *

app = flask.Flask(__name__)
app.secret_key = "COMP247"

data_loader = DataLoader()
clfs = {}


@app.route("/train", methods=["POST"])
def train():
    global clfs
    data_loader.load_data()

    print("Preparing RF model")
    try:
        rf = joblib.load("rf.pkl")
    except:
        rf = RandomForestEmander()
        rf.train(data_loader.X_train_oversampled,
                 data_loader.y_train_oversampled)
        joblib.dump(rf, "rf.pkl")
    clfs["RandomForest-Emander"] = rf
    print("Done")

    print("Preparing HGB model")
    try:
        hgb = joblib.load("hgb.pkl")
    except:
        hgb = HistGradientBoostingWonyoung()
        hgb.train(data_loader.X_train_oversampled,
                  data_loader.y_train_oversampled)
        joblib.dump(hgb, "hgb.pkl")
    clfs["HistGradientBoosting-Wonyoung"] = hgb
    print("Done")

    print("Preparing LR model")
    try:
        lr = joblib.load("lr.pkl")
    except:
        lr = LogisticRegressionUtku()
        lr.train(data_loader.X_train_oversampled,
                 data_loader.y_train_oversampled)
        joblib.dump(lr, "lr.pkl")
    clfs["LogisticRegression-Utku"] = lr
    print("Done")

    print("Preparing SVM model")
    try:
        svm = joblib.load("svm.pkl")
    except:
        svm = SupportVectorClassifierNilkanth()
        svm.train(data_loader.X_train_oversampled,
                  data_loader.y_train_oversampled)
        joblib.dump(svm, "svm.pkl")
    clfs["SVM With Bagging-Nilkanth"] = svm
    print("Done")

    value_by_feature = data_loader.get_unique_values_by_features()
    dump = json.dumps(value_by_feature)
    return dump, 200


@app.route("/predict", methods=["POST"])
def predict():
    json = flask.request.json
    x = pd.DataFrame([json["input_data"]])

    # Convert "NaN" to NaN
    x = x.replace("NaN", float("NaN"))

    x_processed = data_loader.preprocessor.transform(x)

    ret = {}
    for model_name, clf in clfs.items():
        pred = clf.predict(x_processed)[0]
        ret[model_name] = "Non-Fatal" if pred == 1 else "Fatal"

    return ret, 200


@app.route("/test", methods=["POST"])
def test():
    for model_name, clf in clfs.items():
        print("Testing", model_name)
        clf.test(data_loader.X_test, data_loader.y_test)
        print()

    return "Done", 200


@app.route("/")
def index():
    return flask.render_template("index.html")
