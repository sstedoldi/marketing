import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc(y, y_hat):
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


def plot_pr(y, y_hat):
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

def plot_ganancias_calls(df_sem):
    
    gain_cols = [col for col in df_sem.columns if col.startswith('ganancias_')]
    thr_cols = [col for col in df_sem.columns if col.startswith('threshold_')]
    # calls_cols = [col for col in df_sem.columns if col.startswith('total_calls_')]

    # Calcular y graficar las ganancias y umbral promedio
    ganancias_avg = df_sem[gain_cols].mean(axis=1)
    thr_avg = df_sem[thr_cols].mean(axis=1)
    # calls_avg = df_sem[gain_cols].mean(axis=1)

    # Highlight maximum gain
    max_gain_idx = ganancias_avg.idxmax()
    max_gain_call = df_sem.total_calls[max_gain_idx]
    # max_gain_threshold = df_sem.threshold[max_gain_idx]
    max_gain = max(ganancias_avg)
    max_thr = thr_avg[max_gain_idx]
    # max_calls = calls_avg[max_gain_idx]

    fig, ax = plt.subplots()

    # Iterar sobre cada columna de ganancias vs calls
    for sem in df_sem.columns:
        if sem.startswith('ganancias_'):
            ganancias = df_sem[sem]
            calls = df_sem.total_calls
        # if sem.startswith('total_calls_'):
        #     calls = df_sem[sem]
            
            # Plot Threshold vs Calls in gray
            ax.plot(calls, ganancias, color='gray')

    # Plot average in black
    ax.plot(calls, ganancias_avg, label='Ganancias Promedio vs. LLamadas', color='black', linestyle='--')

    ax.scatter(max_gain_call, max_gain, color='red', zorder=5)
    ax.annotate(
        f" Max Gain: {max_gain:.0f}\n Calls: {max_gain_call}\n Threshold: {max_thr:.3f}",
        (max_gain_call, max_gain), 
        textcoords="offset points", 
        xytext=(20, -20),  # Cambiar (0, -20) a (20, -20) para mover hacia la derecha
        ha='left'  # Cambiar alineación horizontal para que el texto se vea bien
    )

    ax.set_xlabel('Llamadas')
    ax.set_ylabel('Ganancias')
    ax.legend()
    plt.title('Ganancias vs Llamadas para cada Semilla')
    plt.show()


def plot_ganancias_thr(df_sem):
    
    gain_cols = [col for col in df_sem.columns if col.startswith('ganancias_')]
    # thr_cols = [col for col in df_sem.columns if col.startswith('threshold_')]
    calls_cols = [col for col in df_sem.columns if col.startswith('total_calls_')]

    # Calcular y graficar las ganancias y umbral promedio
    ganancias_avg = df_sem[gain_cols].mean(axis=1)
    # thr_avg = df_sem[thr_cols].mean(axis=1)
    calls_avg = df_sem[calls_cols].mean(axis=1)

    # Highlight maximum gain
    max_gain_idx = ganancias_avg.idxmax()
    # max_gain_call = df_sem.total_calls[max_gain_idx]
    max_gain_threshold = df_sem.threshold[max_gain_idx]
    max_gain = max(ganancias_avg)
    # max_thr = thr_avg[max_gain_idx]
    max_calls = calls_avg[max_gain_idx]

    fig, ax = plt.subplots()

    # Iterar sobre cada columna de ganancias vs calls
    for sem in df_sem.columns:
        if sem.startswith('ganancias_'):
            ganancias = df_sem[sem]
            # calls = df_sem.total_calls
            thrs = df_sem.threshold

            # Plot Threshold vs Calls in gray
            ax.plot(thrs, ganancias, color='gray')

    # Plot average in black
    ax.plot(thrs, ganancias_avg, label='Ganancias Promedio vs. LLamadas', color='black', linestyle='--')

    ax.scatter(max_gain_threshold, max_gain, color='red', zorder=5)
    ax.annotate(
        f" Max Gain: {max_gain:.0f}\n Calls: {max_calls}\n Threshold: {max_gain_threshold:.3f}",
        (max_gain_threshold, max_gain), 
        textcoords="offset points", 
        xytext=(20, -20),  # Cambiar (0, -20) a (20, -20) para mover hacia la derecha
        ha='left'  # Cambiar alineación horizontal para que el texto se vea bien
    )

    ax.set_xlabel('Umbral')
    ax.set_ylabel('Ganancias')
    ax.legend()
    plt.title('Ganancias vs Umbral para cada Semilla')
    plt.show()