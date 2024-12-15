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
    def __init__(self, data=None, metric='roc_auc', seeds=[1], model_type='decision_tree', n_jobs=-1):
        self.data = data
        self.metric = metric
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

    def plot_roc(self, y, y_hat):
        # Calcular y plotear la curva ROC
        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        print(f"ROC AUC: {roc_auc:.2f}")


    def plot_pr(self, y, y_hat):
        # Calcular y plotear la curva de Precisión-Recall
        precision, recall, thresholds = precision_recall_curve(y, y_hat)
        average_precision = average_precision_score(y, y_hat)

        plt.figure()
        plt.step(recall, precision, color='red', alpha=0.8, where='post', label=f'P-R AUC = {average_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        print(f"P-R AUC: {average_precision:.2f}")


    def evaluate_model(self, model, X, y, plot_roc=False, plot_pr=False):
        # Obtener probabilidades o predicciones
        y_hat = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)

        if plot_roc:
            self.plot_roc(y, y_hat)

        if plot_pr:
            self.plot_pr(y, y_hat)

        if self.metric == 'roc_auc':
            return roc_auc_score(y, y_hat)
        elif self.metric == 'pr-auc':
            return average_precision_score(y, y_hat)
        elif self.metric == 'f1':
            y_pred = model.predict(X)
            return f1_score(y, y_pred, average='macro')
        else:
            raise ValueError(f"Unsupported self.metric: {self.metric}")

    def train_and_evaluate(self, train_index, test_index, X, y, params):
        classifier_class = self.classifier_map[self.model_type]
        model = classifier_class(**params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        score = self.evaluate_model(model, X.iloc[test_index], y.iloc[test_index])
        return model, score

    def optimize_model(self, X, y, storage_name, study_name, test_size=0.3, optimize=True, n_trials=100):
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
                delayed(self.train_and_evaluate)(train_index, test_index, X, y, params)
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

    def plot_comparisons(self, results_base, results_best):
        # Crear DataFrames para cada conjunto de resultados
        df_base = pd.DataFrame({'Score': [result[1] for result in results_base], 'Model': 'Base'})
        df_best = pd.DataFrame({'Score': [result[1] for result in results_best], 'Model': 'Best'})
        df_combined = pd.concat([df_base, df_best])

        # Calcular los promedios
        mean_base = df_base['Score'].mean()
        mean_best = df_best['Score'].mean()

        # Configurar el gráfico de distribuciones superpuestas
        # plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_combined, x='Score', hue='Model', fill=True, alpha=0.4, palette={'Base': 'blue', 'Best': 'red'})

        # Agregar líneas verticales para los promedios
        plt.axvline(mean_base, color='blue', linestyle='--', label=f'Base mean {self.metric}: {mean_base:.2f}')
        plt.axvline(mean_best, color='red', linestyle='--', label=f'Best mean {self.metric}: {mean_best:.2f}')

        # Configurar título y leyenda
        plt.title("Model Performance Comparison")
        plt.xlabel({self.metric})
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Mostrar el gráfico
        plt.show()

        # Imprimir los promedios
        print(f"Base model mean {self.metric}: {mean_base:.2f}")
        print(f"Best model mean {self.metric}: {mean_best:.2f}\n")

    def interpolate_curve(self, x, y, x_grid):
        # Interpola la curva (y en función de x) sobre x_grid
        return np.interp(x_grid, x, y)

    def plot_roc_curves(self, fpr_base_list, tpr_base_list, fpr_best_list, tpr_best_list):
        # Crear una grilla común para FPR
        recall_grid = np.linspace(0, 1, 100)

        # Interpolar todas las curvas base
        tpr_base_interp = [self.interpolate_curve(fpr, tpr, recall_grid) for fpr, tpr in zip(fpr_base_list, tpr_base_list)]
        prec_base_interp = np.mean(tpr_base_interp, axis=0)

        # Interpolar todas las curvas best
        tpr_best_interp = [self.interpolate_curve(fpr, tpr, recall_grid) for fpr, tpr in zip(fpr_best_list, tpr_best_list)]
        prec_best_interp = np.mean(tpr_best_interp, axis=0)

        # Plotear curvas individuales con alpha bajo
        plt.figure()
        for tpr_i in tpr_base_interp:
            plt.plot(recall_grid, tpr_i, color='blue', alpha=0.1)
        for tpr_i in tpr_best_interp:
            plt.plot(recall_grid, tpr_i, color='red', alpha=0.1)

        # Plotear curva promedio con alpha fuerte
        plt.plot(recall_grid, prec_base_interp, color='blue', lw=2,\
                  label=f'Base mean ROC AUC {auc(recall_grid, prec_base_interp):.2f})', alpha=1.0)
        
        plt.plot(recall_grid, prec_best_interp, color='red', lw=2,\
                  label=f'Best mean ROC AUC {auc(recall_grid, prec_best_interp):.2f})', alpha=1.0)

        # Diagonal
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison (All Splits)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def plot_pr_curves(self, prec_base_list, rec_base_list, prec_best_list, rec_best_list, ap_base_splits, ap_best_splits):

        plt.figure()

        # Graficar cada curva Base sin interpolación
        for p, r in zip(prec_base_list, rec_base_list):
            plt.step(r, p, where='post', color='blue', alpha=0.1)

        # Graficar cada curva Best sin interpolación
        for p, r in zip(prec_best_list, rec_best_list):
            plt.step(r, p, where='post', color='red', alpha=0.1)

        # Mostrar AP promedio en el título o en la leyenda
        ap_base_mean = np.mean(ap_base_splits)
        ap_best_mean = np.mean(ap_best_splits)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves (All Splits)')
        # plt.legend(['Base splits', 'Best splits'], loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        print(f'Base mean P-R AUC = {ap_base_mean:.2f}\n Best mean P-R AUC = {ap_best_mean:.2f}')


    def compare_models(self, X, y, folds=5, test_size=0.3):
        print(f"Comparing models with the following configurations:")
        print(f"- Metric: {self.metric}")
        print(f"- Test size: {test_size}")
        print(f"- Base parameters: {self.base_params}")
        print(f"- Best parameters: {self.best_params}")
        print(f"- Number of splits: {folds}")
        print(f"- Random state: {self.seed}")
        print(f"- Number of parallel jobs: {self.n_jobs}")

        sss = StratifiedShuffleSplit(n_splits=folds, test_size=test_size, 
                                     random_state=self.seed)

        results_base = []
        results_best = []

        # Listas para guardar curvas y métricas
        fpr_base_list, tpr_base_list = [], []
        fpr_best_list, tpr_best_list = [], []
        prec_base_list, rec_base_list = [], []
        prec_best_list, rec_best_list = [], []
        ap_base_splits, ap_best_splits = [], []

        classifier_class = self.classifier_map[self.model_type]

        for train_index, test_index in sss.split(X, y):
            X_train_split, X_test_split = X.iloc[train_index], X.iloc[test_index]
            y_train_split, y_test_split = y.iloc[train_index], y.iloc[test_index]

            # Base model training and evaluation
            base_model = classifier_class(**self.base_params)
            base_model.fit(X_train_split, y_train_split)
            score_base = self.evaluate_model(base_model, X_test_split, y_test_split)
            results_base.append((base_model, score_base))

            # Best model training and evaluation
            if self.best_params is None:
                raise ValueError("Best parameters not found. Run optimize_model first.")
            best_model = classifier_class(**self.best_params)
            best_model.fit(X_train_split, y_train_split)
            score_best = self.evaluate_model(best_model, X_test_split, y_test_split)
            results_best.append((best_model, score_best))

            # Obtener probabilidades
            y_hat_base = base_model.predict_proba(X_test_split)[:, 1]\
                         if hasattr(base_model, 'predict_proba')\
                         else base_model.predict(X_test_split)
            
            y_hat_best = best_model.predict_proba(X_test_split)[:, 1]\
                         if hasattr(best_model, 'predict_proba')\
                         else base_model.predict(X_test_split)

            # Calcular curvas ROC
            fpr_b, tpr_b, _ = roc_curve(y_test_split, y_hat_base)
            fpr_base_list.append(fpr_b)
            tpr_base_list.append(tpr_b)

            fpr_bm, tpr_bm, _ = roc_curve(y_test_split, y_hat_best)
            fpr_best_list.append(fpr_bm)
            tpr_best_list.append(tpr_bm)

            # Calcular curvas PR
            # prec_b, rec_b, _ = precision_recall_fixed_grid(y_test_split, y_hat_base)
            prec_b, rec_b, _ = precision_recall_curve(y_test_split, y_hat_base)
            prec_base_list.append(prec_b)
            rec_base_list.append(rec_b)

            # prec_bm, rec_bm, _ = precision_recall_fixed_grid(y_test_split, y_hat_best)
            prec_bm, rec_bm, _ = precision_recall_curve(y_test_split, y_hat_best)
            prec_best_list.append(prec_bm)
            rec_best_list.append(rec_bm)

            # Calcular AP para cada split
            ap_base = average_precision_score(y_test_split, y_hat_base)
            ap_best = average_precision_score(y_test_split, y_hat_best)
            ap_base_splits.append(ap_base)
            ap_best_splits.append(ap_best)

        # Graficar las curvas ROC promedio y todas las curvas
        self.plot_roc_curves(fpr_base_list, tpr_base_list, fpr_best_list, tpr_best_list)

        # Graficar las curvas PR promedio y todas las curvas
        self.plot_pr_curves(prec_base_list, rec_base_list, prec_best_list, rec_best_list,\
                             ap_base_splits, ap_best_splits)

        # Boxplot comparativo
        self.plot_comparisons(results_base, results_best)

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
