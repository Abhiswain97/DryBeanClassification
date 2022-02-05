import numpy as np
import seaborn as sns

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn import metrics

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_score,
)

from imblearn.metrics import classification_report_imbalanced

import optuna

sns.set(rc={"figure.figsize": (10, 8)}, font_scale=1.25)

np.random.seed(42)


class Modelling:
    def __init__(self, X_train, y_train, X_test, y_test, model, class_names):
        self._X_train, self._y_train, self._X_test, self._y_test = (
            X_train,
            y_train,
            X_test,
            y_test,
        )
        self._model = model
        self._tuned_model = None
        self.class_names = class_names
        self.scorer = metrics.make_scorer(metrics.f1_score, average="weighted")
        self.preds = None
        self.study = None
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self._X_train)
        self.X_test_scaled = self.scaler.transform(self._X_test)

        print(f"Using model: {self._model.__class__.__name__}")

    def get_preds(self):
        """
        This method calculates the the predictions either
        using base-line or tuned-model

        The sequence of calling `perform_cross_validation()`
        or `tune()` method before this depends on how it's
        going to calculate the predsictions


        Parameters
        ----------

        self.preds: the np.array of predictions

        If called after `perform_cross_validation()` uses baseline model
        If called after `tune()` used tuned_model

        """
        if self._tuned_model is not None:
            if self._tuned_model.__class__.__name__ == "LinearSVC":
                self.preds = self._tuned_model.predict(self.X_test_scaled)
            else:
                self.preds = self._tuned_model.predict(self._X_test)
        else:
            if self._model.__class__.__name__ == "LinearSVC":
                self.preds = self._model.predict(self.X_test_scaled)
            else:
                self.preds = self._model.predict(self._X_test)

    def perform_cross_validation(self):
        """
        Performs 10-fold cross-validation over the baseline model
        """
        print("Performing 10-fold cross validation")

        if self._model.__class__.__name__ == "LinearSVC":
            results = cross_validate(
                estimator=self._model,
                X=self.X_train_scaled,
                y=self._y_train,
                scoring=self.scorer,
                cv=StratifiedKFold(n_splits=10),
                verbose=2,
                n_jobs=-1,
                return_estimator=True,
            )
            self._model = results["estimator"][-1]
        else:
            results = cross_validate(
                estimator=self._model,
                X=self._X_train,
                y=self._y_train,
                scoring=self.scorer,
                cv=StratifiedKFold(n_splits=10),
                verbose=2,
                n_jobs=-1,
                return_estimator=True,
            )
            self._model = results["estimator"][-1]

        # Calling the get_preds() function after cross-val
        # This should use base-line model for metrics and confusion-matrix
        self.get_preds()

    def plot_cf(self, save=True):
        """
        Plots the confusion matrix

        Depending on the state of tuned_model this function uses
        tuned_model or base-line model for plotting the confusion matrix.
        This behavior is controlled by `get_preds()` method.
        The sequence of model-fitting and calling get_preds defines whether
        the confusion matrix is made from base-line or tuned-model. Refer `get_preds()`

        Parameters
        ----------

        save: default=True
            To the save the cf as a .png file
        """

        if self._tuned_model is not None:
            cf = metrics.confusion_matrix(self._y_test, self.preds)
        else:
            cf = metrics.confusion_matrix(self._y_test, self.preds)

        ax = sns.heatmap(
            cf,
            annot=True,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            fmt="d",
            cmap="viridis_r",
        )

        title = str()

        if self._tuned_model is not None:
            title = f"Confusion Matrix for Tuned {self._model.__class__.__name__}"
            ax.set_title(title, fontsize="30")
        else:
            title = f"Confusion Matrix for Base-line {self._model.__class__.__name__}"
            ax.set_title(title, fontsize="30")

        ax.set_ylabel("Actual Class", fontsize="20")
        _ = ax.set_xlabel("Predicted Class", fontsize="20")

        if save:
            fig = ax.get_figure()
            fig.savefig(f"./ML_results/{title}.png", bbox_inches="tight")
            print(f"Figure saved at ./ML_results/{title}.png")

    def get_metrics(self):
        """
        This method prints the `classification_report_imbalanced()`

        Depending on the state of tuned_model this function uses
        tuned_model or base-line model calculating the metrics.
        This behavior is controlled by `get_preds()` method.
        The sequence of model-fitting and calling get_preds defines whether
        the metrics are calculated from base-line or tuned-model. Refer `get_preds()`
        """

        print(
            classification_report_imbalanced(
                self._y_test, self.preds, target_names=self.class_names
            )
        )

    def objective(self, trial):

        classifier_name = trial.suggest_categorical("clf_name", ["rf", "dt", "svm"])

        if classifier_name == "rf":
            params = {
                "max_depth": trial.suggest_categorical(
                    "rf_max_depth", [10, 20, 30, 40]
                ),
                "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
                "max_features": trial.suggest_categorical(
                    "rf_max_features", ["auto", "sqrt"]
                ),
                "class_weight": trial.suggest_categorical(
                    "rf_class_weight", ["balanced", "balanced_subsample"]
                ),
                "n_jobs": -1,
            }
            self._model = RandomForestClassifier(**params)

            return np.mean(
                cross_val_score(
                    estimator=self._model,
                    X=self._X_train,
                    y=self._y_train,
                    scoring=self.scorer,
                    n_jobs=-1,
                )
            )

        elif classifier_name == "dt":
            params = {
                "max_depth": trial.suggest_categorical(
                    "dt_max_depth", [2, 3, 5, 10, 20]
                ),
                "min_samples_leaf": trial.suggest_categorical(
                    "dt_min_samples_leaf", [5, 10, 20, 50, 100]
                ),
                "max_features": trial.suggest_categorical(
                    "dt_max_features", ["sqrt", "auto", "log2"]
                ),
                "criterion": trial.suggest_categorical(
                    "dt_criterion", ["gini", "entropy"]
                ),
                "splitter": trial.suggest_categorical(
                    "dt_splitter", ["best", "random"]
                ),
            }
            self._model = DecisionTreeClassifier(**params)

            return np.mean(
                cross_val_score(
                    estimator=self._model,
                    X=self._X_train,
                    y=self._y_train,
                    scoring=self.scorer,
                    n_jobs=-1,
                )
            )

        elif classifier_name == "svm":
            params = {
                "C": trial.suggest_float("svm_C", 0.1, 3.0, log=True),
                "loss": trial.suggest_categorical(
                    "svm_loss", ["hinge", "squared_hinge"]
                ),
                "class_weight": "balanced",
            }
            self._model = LinearSVC()

            return np.mean(
                cross_val_score(
                    estimator=self._model,
                    X=self.X_train_scaled,
                    y=self._y_train,
                    scoring=self.scorer,
                    n_jobs=-1,
                )
            )

    def clean_hparams(self):
        d = {}
        for k, v in self.study.best_params.items():
            if k == "clf_name":
                continue
            idx = k.index("_") + 1
            k = k[idx:]
            d[k] = v

        return d

    def tune(self, n_trials=50):

        # Create Optuna study 'DryBeanStudy'
        self.study = optuna.create_study(
            direction="maximize", study_name="DryBeanStudy"
        )

        # Optimize the study
        self.study.optimize(self.objective, n_trials=n_trials)

        best_params = self.clean_hparams()

        if self.study.best_params["clf_name"] == "rf":
            self._tuned_model = RandomForestClassifier(**best_params).fit(
                self._X_train, self._y_train
            )
        elif self.study.best_params["clf_name"] == "dt":
            self._tuned_model = DecisionTreeClassifier(**best_params).fit(
                self._X_train, self._y_train
            )
        else:
            self._tuned_model = LinearSVC(**best_params).fit(
                self.X_train_scaled, self._y_train
            )

        # Calling the get_preds() function after cross-val
        # This should update Tuned model for metrics and confusion-matrix
        self.get_preds()

        return self.study
