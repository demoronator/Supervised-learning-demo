import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC


class RandomForestEmander:
    """
    Random Forest Classifier - Emander
    """

    def train(self, X_train, y_train):
        rfc = RandomForestClassifier(n_estimators=200, random_state=13)

        # define the hyperparameter search space
        param_dist = {"n_estimators": range(10, 300, 20),
                      "max_depth": range(1, 30, 2),
                      "max_features": ["sqrt", "log2", None],
                      "min_samples_split": range(2, 15, 3),
                      "min_samples_leaf": range(1, 15, 3)}

        # perform randomized grid search
        random_search = RandomizedSearchCV(
            rfc,
            param_distributions=param_dist,
            scoring="accuracy",
            cv=10,
            n_iter=25,
            refit=True,
            verbose=3,
            random_state=15,
            n_jobs=-2,
        )

        # fit the search to the data
        random_search.fit(X_train, y_train)

        # print the best hyperparameters and associated score
        print("Best hyperparameters:", random_search.best_params_)
        print("Best score:", random_search.best_score_)
        print("Best Estimator:", random_search.best_estimator_)

        self.best_estimator: RandomForestClassifier = random_search.best_estimator_

    def test(self, X_test, y_test):
        # ### Random Forest Classifier Testing the model - Emander
        rfc_predict_fine_tuned = self.best_estimator.predict(X_test)

        print("Accuracy score:", accuracy_score(
            y_test, rfc_predict_fine_tuned))
        print("Precision:", precision_score(y_test, rfc_predict_fine_tuned))
        print("Recall:", recall_score(y_test, rfc_predict_fine_tuned))
        print("F1 Score:", f1_score(y_test, rfc_predict_fine_tuned))
        print("Confusion Matrix:\n", confusion_matrix(
            y_test, rfc_predict_fine_tuned))

    def predict(self, X):
        return self.best_estimator.predict(X)

    def predict_proba(self, X):
        return self.best_estimator.predict_proba(X)


class HistGradientBoostingWonyoung:
    """
    Histogram-based Gradient Boosting Classifier - Wonyoung
    """

    def train(self, X_train, y_train):
        seed_w = 301215136 % 100

        # Perform random search for hyperparameters
        param_dist = {
            "learning_rate": np.linspace(0.01, 1),
            "l2_regularization": np.linspace(0, 1),
        }

        random_search = RandomizedSearchCV(
            HistGradientBoostingClassifier(
                # Specify the loss function for imbalanced data
                scoring="average_precision",
                max_iter=1000,
                random_state=seed_w,
            ),
            param_distributions=param_dist,
            n_iter=20, cv=5, random_state=seed_w,
            n_jobs=-2,
            verbose=3,
        )
        random_search.fit(X_train, y_train)
        self.hgb: HistGradientBoostingClassifier = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")

    def test(self, X_test, y_test):
        # Evaluate classifier performance on test set
        y_pred = self.hgb.predict(X_test)
        print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("Precision: {}".format(precision_score(
            y_test, y_pred, average="weighted")))
        print("Recall: {}".format(recall_score(
            y_test, y_pred, average="weighted")))
        print("F1: {}".format(f1_score(
            y_test, y_pred, average="weighted")))
        print("Confusion matrix:\n{}".format(
            confusion_matrix(y_test, y_pred)))

    def predict(self, X):
        return self.hgb.predict(X)

    def predict_proba(self, X):
        return self.hgb.predict_proba(X)


class LogisticRegressionUtku:
    """
    Logistic Regression Model - Utku Emecan
    """

    def train(self, X_train, y_train):
        logreg = LogisticRegression(max_iter=100000)
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1"],
            "solver": ["saga"],
        }

        grid_search = GridSearchCV(
            logreg, param_grid, cv=10, n_jobs=-2, verbose=3)
        grid_search.fit(X_train, y_train)

        print("Best parameters found using GridSearchCV: ",
              grid_search.best_params_)

        random_search = RandomizedSearchCV(
            logreg, param_grid, n_iter=10, cv=10, n_jobs=-2)
        random_search.fit(X_train, y_train)

        print("Best parameters found using RandomizedSearchCV: ",
              random_search.best_params_)

        self.best_logreg: LogisticRegression = random_search.best_estimator_

    def test(self, X_test, y_test):
        y_pred = self.best_logreg.predict(X_test)
        print("Classification report:\n", classification_report(y_test, y_pred))
        print("Accuracy score: ", accuracy_score(y_test, y_pred))

    def predict(self, X):
        return self.best_logreg.predict(X)

    def predict_proba(self, X):
        return self.best_logreg.predict_proba(X)


class SupportVectorClassifierNilkanth:
    """
    Support Vector Classifier - Nilkanth
    """

    def train(self, X_train, y_train):
        # define base classifier
        base_svm = SVC(kernel="linear")
        # create an ensemble of SVM classifiers using bagging
        ensemble_svm = BaggingClassifier(
            estimator=base_svm, n_estimators=10, random_state=64)

        param_grid = {
            "estimator__C": [0.1, 1, 10],
            # "n_estimators": [5, 10, 15],
            # "max_samples": [0.5, 0.7, 1.0],
            "max_features": [0.5]  # [0.5, 0.7, 1.0],
        }

        grid = GridSearchCV(ensemble_svm, param_grid, verbose=3, n_jobs=-2)

        # train the ensemble SVM classifier on the training Nilkanth_data
        grid.fit(X_train, y_train)
        # Print the best parameters and score
        print("Best parameters: ", grid.best_params_)
        print("Best score: ", grid.best_score_)
        # make predictions on the test Nilkanth_data
        # Make predictions on the test data using the best model
        self.best_model: SVC = grid.best_estimator_

    def test(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)

        # Evaluate the performance of the classifier
        report = classification_report(y_test, y_pred)
        print(report)
        print("Accuracy score: ", accuracy_score(y_test, y_pred))

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)
