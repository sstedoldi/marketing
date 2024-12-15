import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from joblib import Parallel, delayed
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

class OptiPipeline:
    def __init__(self, data=None, seeds=[1], model_type='decision_tree', n_jobs=-1):
        self.data = data
        self.seeds = seeds
        self.seed = seeds[0]
        self.n_jobs = int(n_jobs)
        self.model_type = model_type
        self.models = {}
        self.base_params = {'random_state': self.seed}
        self.best_params = None
        self.base_model = None
        self.best_model = None

        # Map model_type to corresponding classifier
        self.classifier_map = {
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }

    # def def_xy(self, target='clase_ternaria'):
    #     if self.data is None:
    #         raise ValueError("Data cannot be None. Provide a valid dataset.")

    #     X = self.data.drop(columns=[target])
    #     y = self.data[target]
    #     return X, y

    # def evaluate_model(self, model, X, y, metric='roc_auc', plot_roc=False):
    #     y_hat = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
    #     if metric == 'roc_auc':
    #         if plot_roc:
    #             fpr, tpr, _ = roc_curve(y, y_hat)
    #             roc_auc = auc(fpr, tpr)
    #             plt.figure()
    #             plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    #             plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    #             plt.xlabel('False Positive Rate')
    #             plt.ylabel('True Positive Rate')
    #             plt.title('Receiver Operating Characteristic')
    #             plt.legend(loc="lower right")
    #             plt.show()
    #         return roc_auc_score(y, y_hat)
    #     elif metric == 'f1':
    #         y_pred = model.predict(X)
    #         return f1_score(y, y_pred, average='macro')
    #     else:
    #         raise ValueError(f"Unsupported metric: {metric}")

    def plot_roc(self, y, y_hat):
        # Calcular y plotear la curva ROC
        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)

        # # Calcular precisión y recall reales para cada threshold
        # precision = tpr / (tpr + fpr + 1e-9)  # Evitar división por cero
        # recall = tpr
        # f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-9) for i in range(len(tpr))]
        # optimal_idx = np.argmax(f1_scores)
        # optimal_threshold = thresholds[optimal_idx]

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        # plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal F1 = {f1_scores[optimal_idx]:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # print("--- ROC Metrics ---")
        print(f"ROC AUC: {roc_auc:.2f}")
        # print(f"Optimal Threshold (ROC): {optimal_threshold:.2f}")
        # print(f"Optimal F1 (ROC): {f1_scores[optimal_idx]:.2f}")
        # print(f"Precision at Optimal Threshold: {precision[optimal_idx]:.2f}")
        # print(f"Recall at Optimal Threshold: {recall[optimal_idx]:.2f}")

    def plot_pr(self, y, y_hat):
        # Calcular y plotear la curva de Precisión-Recall
        precision, recall, thresholds = precision_recall_curve(y, y_hat)
        average_precision = average_precision_score(y, y_hat)

        # # Calcular el threshold óptimo basado en el F1
        # f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-9) for i in range(len(precision))]
        # optimal_idx = np.argmax(f1_scores)
        # optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

        plt.figure()
        plt.step(recall, precision, color='red', alpha=0.8, where='post', label=f'AP = {average_precision:.2f}')
        # plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', label=f'Optimal F1 = {f1_scores[optimal_idx]:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower left")
        plt.show()

        # print("--- Precision-Recall Metrics ---")
        print(f"P-R AUC: {average_precision:.2f}")
        # print(f"Optimal Threshold (PR): {optimal_threshold:.2f}")
        # print(f"Optimal F1 (PR): {f1_scores[optimal_idx]:.2f}")
        # print(f"Precision at Optimal Threshold: {precision[optimal_idx]:.2f}")
        # print(f"Recall at Optimal Threshold: {recall[optimal_idx]:.2f}")

    def evaluate_model(self, model, X, y, metric='roc_auc', plot_roc=False, plot_pr=False):
        # Obtener probabilidades o predicciones
        y_hat = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)

        if plot_roc:
            self.plot_roc(y, y_hat)

        if plot_pr:
            self.plot_pr(y, y_hat)

        if metric == 'roc_auc':
            return roc_auc_score(y, y_hat)
        elif metric == 'f1':
            y_pred = model.predict(X)
            return f1_score(y, y_pred, average='macro')
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def train_and_evaluate(self, train_index, test_index, X, y, params, metric):
        classifier_class = self.classifier_map[self.model_type]
        model = classifier_class(**params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        score = self.evaluate_model(model, X.iloc[test_index], y.iloc[test_index], metric=metric)
        return model, score

    def optimize_model(self, X, y, storage_name, study_name, test_size=0.3, optimize=True, n_trials=100, metric='roc_auc'):
        sss = ShuffleSplit(n_splits=5, test_size=test_size, random_state=self.seed)

        def objective(trial):
            if self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                }
            elif self.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                }
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.train_and_evaluate)(train_index, test_index, X, y, params, metric)
                for train_index, test_index in sss.split(X, y)
            )

            return np.mean([result[1] for result in results])

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True
        )

        if optimize:
            print(f"Optimizing {self.model_type} with {n_trials} trials")
            study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_trial.params
        print(f"Best parameters for {self.model_type}: {self.best_params}")
        return self.best_params

    def train_base_model(self, X_train, y_train):
        classifier_class = self.classifier_map[self.model_type]
        self.base_model = classifier_class(**self.base_params)
        self.base_model.fit(X_train, y_train)

    def train_best_model(self, X_train, y_train):
        if self.best_params is None:
            raise ValueError("Best parameters not found. Run optimize_model first.")
        classifier_class = self.classifier_map[self.model_type]
        self.best_model = classifier_class(**self.best_params)
        self.best_model.fit(X_train, y_train)

    def plot_comparisons(self, X, y, results_base, results_best):
        df_base = pd.DataFrame({'Score': [result[1] for result in results_base], 'Model': 'Base'})
        df_best = pd.DataFrame({'Score': [result[1] for result in results_best], 'Model': 'Best'})
        df_combined = pd.concat([df_base, df_best])

        sns.boxplot(x='Model', y='Score', data=df_combined)
        plt.title("Model Performance Comparison")
        plt.show()

        print(f"Base model mean score: {df_base['Score'].mean()}")
        print(f"Best model mean score: {df_best['Score'].mean()}")

    def compare_models(self, X, y, folds=5, test_size=0.3, metric='roc_auc'):
        print(f"Comparing models with the following configurations:")
        print(f"- Metric: {metric}")
        print(f"- Test size: {test_size}")
        print(f"- Base parameters: {self.base_params}")
        print(f"- Best parameters: {self.best_params}")
        print(f"- Number of splits: {folds}")
        print(f"- Random state: {self.seed}")
        print(f"- Number of parallel jobs: {self.n_jobs}")

        sss = StratifiedShuffleSplit(n_splits=folds, test_size=test_size, random_state=self.seed)

        results_base = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_and_evaluate)(train_index, test_index, X, y, self.base_params, metric)
            for train_index, test_index in sss.split(X, y)
        )

        results_best = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_and_evaluate)(train_index, test_index, X, y, self.best_params, metric)
            for train_index, test_index in sss.split(X, y)
        )

        self.plot_comparisons(X, y, results_base, results_best)

        return results_base, results_best


def analyze_study(db_path, study_name):
    # Load the study from the database
    study = optuna.load_study(study_name=study_name, storage=db_path)

    # Extract all trial data as a DataFrame
    df = study.trials_dataframe()
    print("\nStudy Trials DataFrame:\n")
    print(df.head(10).to_markdown())

    # Display basic statistics about the trials
    print("\nBasic Statistics of Trials:\n")
    print(df.describe().to_markdown())

    # Identify the best trial
    best_trial = study.best_trial
    print("\nBest Trial:\n")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return study
