import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summary_statistics(data):
    return data.describe().T.to_markdown()

def summary_statistics_comparative(data1, data2, key1='1', key2='2', stats=None):
    summary1 = data1.describe().T
    summary2 = data2.describe().T

    # Filtrar las columnas de estadísticas si se especifica
    if stats is not None:
        summary1 = summary1[stats]
        summary2 = summary2[stats]

    summary_combined = pd.concat([summary1, summary2], axis=1, keys=[key1, key2])

    # Obtener la lista de estadísticas seleccionadas
    selected_stats = summary1.columns.tolist()

    new_columns = []
    for stat in selected_stats:
        new_columns.extend([(key1, stat), (key2, stat)])

    summary_combined = summary_combined.reindex(columns=new_columns)

    return summary_combined

def is_int(data, feature):
    if pd.api.types.is_numeric_dtype(data[feature]):
        if pd.api.types.is_integer_dtype(data[feature]):
            return True
        else:
            return False

def plot_distributions(data, target, title_suffix='', sample_frac=0.2, transform='log1p', to_log=[]):
    """Plot distributions of numerical and categorical features, determining if they are discrete or continuous,
    and sample the dataset to speed up the process. Apply appropriate transformation to continuous variables."""

    data_sampled = data.sample(frac=sample_frac, random_state=42)

    num_features = data_sampled.drop([target], axis=1).select_dtypes(include=np.number).columns.tolist()

    cat_features = data_sampled.drop(num_features+[target], axis=1).columns.tolist()

    target_classes = data_sampled[target].unique()

    # Binary palette
    palette = {target_classes[1]: sns.color_palette()[0], target_classes[0]: sns.color_palette()[2]}

    print('Variables numéricas')
    for feature in num_features:
        plt.figure(figsize=(10, 4))

        # Determine if the feature is discrete or continuous
        if is_int(data_sampled, feature):

            # Lista para almacenar los DataFrames de frecuencias relativas
            freq_data_list = []

            # Calculamos la frecuencia relativa de feature para cada clase del target
            for t_class in target_classes:
                # Filtramos los datos por la clase del target
                data_tclass = data_sampled[data_sampled[target] == t_class]

                # Calculamos la frecuencia relativa de cada categoría de feature
                counts = data_tclass[feature].value_counts(normalize=True).reset_index()
                counts.columns = [feature, 'relative_freq']
                counts[target] = t_class  # Añadimos la clase del target

                # Añadimos al listado
                freq_data_list.append(counts)

            # Concatenamos los DataFrames
            freq_data = pd.concat(freq_data_list, ignore_index=True)

            # Graficamos las frecuencias relativas
            sns.barplot(x=feature, y='relative_freq', hue=target, data=freq_data, palette=palette)

            # Añadimos etiquetas y título
            plt.title(f'Relative Frequency Plot of {feature} {title_suffix}')
            plt.ylabel('Relative Frequency')

            plt.show()

        else:
            if feature in to_log:
                # Handle NaN and negative values before applying log1p
                if transform == 'log1p':
                    positive_values = data_sampled[feature] > 0
                    transformed_feature = np.log1p(data_sampled[feature].where(positive_values))
                    title_t = f'Log1p-Transformed Distribution of {feature} {title_suffix}'
                    x_label = f'Log1p({feature})'
                elif transform == 'sqrt':
                    non_negative_values = data_sampled[feature] - data_sampled[feature].min() + 1
                    transformed_feature = np.sqrt(non_negative_values)
                    title_t = f'Square Root Transformed Distribution of {feature} {title_suffix}'
                    x_label = f'Square Root({feature})'

                # Normal plot
                title = f'Distribution of {feature} {title_suffix}'
                df_ = pd.DataFrame({feature: data_sampled[feature], target: data_sampled[target]})
                sns.histplot(data=df_, x=feature, hue=target, kde=True, element='step', stat="probability", 
                            common_norm=False, palette=palette)
                plt.title(title)

                plt.show()

                # Transformed plot
                df_transformed = pd.DataFrame({feature: transformed_feature, target: data_sampled[target]})
                sns.histplot(data=df_transformed, x=feature, hue=target, kde=True, element='step', stat="probability", 
                            common_norm=False, palette=palette)
                plt.title(title_t)
                plt.xlabel(x_label)

                plt.show()

            else:
                title = f'Distribution of {feature} {title_suffix}'
                df_ = pd.DataFrame({feature: data_sampled[feature], target: data_sampled[target]})
                sns.histplot(data=df_, x=feature, hue=target, kde=True, element='step', stat="probability", 
                            common_norm=False, palette=palette)
                plt.title(title)

                plt.show()

    print('Variables categóricas o booleanas')
    # Graficar variables categóricas o booleanas
    for feature in cat_features:
        
        plt.figure(figsize=(10, 4))

        # Lista para almacenar los DataFrames de frecuencias relativas
        freq_data_list = []

        # Calculamos la frecuencia relativa de feature para cada clase del target
        for t_class in target_classes:
            # Filtramos los datos por la clase del target
            data_tclass = data_sampled[data_sampled[target] == t_class]

            # Calculamos la frecuencia relativa de cada categoría de feature
            counts = data_tclass[feature].value_counts(normalize=True).reset_index()
            counts.columns = [feature, 'relative_freq']
            counts[target] = t_class  # Añadimos la clase del target

            # Añadimos al listado
            freq_data_list.append(counts)

        # Concatenamos los DataFrames
        freq_data = pd.concat(freq_data_list, ignore_index=True)

        # Graficamos las frecuencias relativas
        sns.barplot(x=feature, y='relative_freq', hue=target, data=freq_data, palette=palette)

        # Rotar etiquetas del eje x
        plt.xticks(rotation=45, ha='right')

        # Añadimos etiquetas y título
        plt.title(f'Relative Frequency Plot of {feature} {title_suffix}')
        plt.ylabel('Relative Frequency')

        plt.show()

def plot_distributions_by_cluster(data, clusters_dict, 
                                  target='cluster', top_n=5, 
                                  sample_frac=0.2, 
                                  transform='log1p', to_log=[],
                                  cluster_stats=None,
                                  variable_config=None):

    data_sampled = data.sample(frac=sample_frac, random_state=42)
    
    i = 0  # Para asignar colores
    
    # Recorrer los clusters y sus variables relevantes
    for cluster, features in clusters_dict.items():

        print(f"\nComparando Cluster {cluster} contra el resto")

        if cluster_stats is not None:
            cluster_info = cluster_stats.loc[cluster_stats[target] == cluster].copy()
            cluster_info['Casos %'] = 100*(cluster_info['venta_c'] + cluster_info['no-venta_c'])/len(data)
            cluster_info['Ventas %'] = 100*cluster_info['venta_p']
            cluster_info['No-ventas %'] = 100*cluster_info['no-venta_p']
            cluster_info['Ventas sobre el total %'] = 100*cluster_info['venta_c']/cluster_stats['venta_c'].sum()

            print(cluster_info[['Casos %','Ventas %','No-ventas %','Ventas sobre el total %']].to_markdown())
        
        # Tomar las N variables más relevantes para este cluster
        relevant_features = features[:top_n]
        
        # **Crear una copia temporal del DataFrame para este cluster**
        data_for_plot = data_sampled.copy()
        
        # # **Aplicar transformaciones solo a las variables relevantes**
        # if cluster != 'XXXX': # para cluster quasi-negativo
        #     if variable_config is not None:
        #         # Variables a convertir a float
        #         vars_to_float = [var for var in relevant_features if var in variable_config.to_float]
        #         data_for_plot = convert_to_float(data_for_plot, vars_to_float)
                
        #         # Variables a convertir a int
        #         vars_to_int = [var for var in relevant_features if var in variable_config.to_int]
        #         data_for_plot = convert_to_int(data_for_plot, vars_to_int)
                
        #         # Variables a recortar high 5%
        #         vars_cut_high_5 = [var for var in relevant_features if var in variable_config.to_cut_high_5]
        #         data_for_plot = cut_high_5(data_for_plot, vars_cut_high_5)
                
        #         # Variables a recortar high 10%
        #         vars_cut_high_10 = [var for var in relevant_features if var in variable_config.to_cut_high_10]
        #         data_for_plot = cut_high_10(data_for_plot, vars_cut_high_10)
                
        #         # Variables a recortar low 5%
        #         vars_cut_low_5 = [var for var in relevant_features if var in variable_config.to_cut_low_5]
        #         data_for_plot = cut_low_5(data_for_plot, vars_cut_low_5)
                
        #         # Variables a recortar low 10%
        #         vars_cut_low_10 = [var for var in relevant_features if var in variable_config.to_cut_low_10]
        #         data_for_plot = cut_low_10(data_for_plot, vars_cut_low_10)
                
        #         # Variables a recortar low-high (5% y 95%)
        #         vars_cut_low_high = [var for var in relevant_features if var in variable_config.to_cut_low_high]
        #         data_for_plot = cut_low_high(data_for_plot, vars_cut_low_high)
        
        # Crear una columna nueva que identifique si es el cluster actual o el resto
        data_for_plot['comparison'] = data_for_plot[target].apply(lambda x: f'{cluster}' if x == cluster else 'Resto')
        
        # colores
        i = i+1 if i < 9 else 1

        for feature in relevant_features:
            plt.figure(figsize=(10, 4))
            palette = {f'{cluster}': sns.color_palette()[i], 'Resto': sns.color_palette()[0]}
            
            # Determinar si la variable es discreta o continua
            if is_int(data_for_plot, feature):
                # Graficar distribución de frecuencias relativas para variable discreta
                sns.histplot(data=data_for_plot, x=feature, hue='comparison', multiple="dodge", stat="probability", 
                            common_norm=False, palette=palette)
                plt.title(f'Distribución de {feature} ({cluster} vs Resto)')
                plt.ylabel('Frecuencia relativa')
                plt.show()
            
            # else: # analisis sin transformaciones
                # # Si es continua, aplicar transformaciones si es necesario
                # if feature in variable_config.to_log and transform in ['log1p', 'sqrt']:
                #     # Aplicar transformación
                #     if transform == 'log1p':
                #         positive_values = data_for_plot[feature] > 0
                #         data_for_plot = data_for_plot[positive_values]  # Filtrar valores positivos
                #         transformed_feature = np.log1p(data_for_plot[feature])
                #         title_t = f'Distribución Transformada Log1p de {feature} ({cluster} vs Resto)'
                #         x_label = f'Log1p({feature})'
                #     elif transform == 'sqrt':
                #         non_negative_values = data_for_plot[feature] - data_for_plot[feature].min() + 1
                #         transformed_feature = np.sqrt(non_negative_values)
                #         title_t = f'Distribución Transformada Raíz Cuadrada de {feature} ({cluster} vs Resto)'
                #         x_label = f'Sqrt({feature})'
                    
                #     # Plotear la distribución transformada
                #     df_transformed = pd.DataFrame({
                #         feature: transformed_feature, 
                #         'comparison': data_for_plot['comparison']
                #     })
                #     sns.histplot(data=df_transformed, x=feature, hue='comparison', kde=True, element='step', stat="probability", 
                #                 common_norm=False, palette=palette)
                #     plt.title(title_t)
                #     plt.xlabel(x_label)
                #     plt.show()
                
            else:
                # Graficar distribución normal si no se necesita transformación
                sns.histplot(data=data_for_plot, x=feature, hue='comparison', kde=True, element='step', stat="probability", 
                            common_norm=False, palette=palette)
                plt.title(f'Distribución de {feature} ({cluster} vs Resto)')
                plt.show()

def distanceMatrix(model, X):
    """
    Calcula una matriz de distancias entre muestras basándose en cuántas veces caen en las mismas hojas
    de un bosque aleatorio (Random Forest).

    Parámetros:
        model: RandomForestClassifier o RandomForestRegressor entrenado.
        X: numpy array o pandas DataFrame con las características de entrada.

    Retorna:
        Una matriz de distancias entre las muestras de X.
    """
    # Obtener las hojas (nodos terminales) donde caen las muestras para cada árbol del bosque.
    # La matriz 'terminals' tiene forma (n_samples, n_trees).
    terminals = model.apply(X)
    
    # Obtener el número de árboles en el bosque.
    nTrees = terminals.shape[1]
    
    # Inicializar la matriz de proximidad usando el primer árbol.
    # 'a' contiene los índices de las hojas donde caen las muestras para el primer árbol.
    a = terminals[:, 0]
    
    # Crear la matriz de similitud para el primer árbol.
    # np.equal.outer(a, a) compara cada par de muestras (i, j) para determinar si caen en la misma hoja.
    # Multiplicamos por 1 para convertir los valores booleanos en enteros (1 si son iguales, 0 si no lo son).
    proxMat = 1 * np.equal.outer(a, a)
    
    # Iterar sobre el resto de los árboles (del segundo al último) para acumular la matriz de proximidad.
    for i in range(1, nTrees):
        # Obtener las hojas donde caen las muestras para el árbol actual.
        a = terminals[:, i]
        
        # Sumar la matriz de similitud del árbol actual a la matriz de proximidad acumulada.
        proxMat += 1 * np.equal.outer(a, a)
    
    # Normalizar la matriz de proximidad dividiendo por el número total de árboles.
    # Esto genera una matriz con valores entre 0 y 1, donde 1 indica que dos muestras siempre
    # caen juntas en las mismas hojas y 0 que nunca lo hacen.
    proxMat = proxMat / nTrees
    
    # Convertir la matriz de proximidad en una matriz de distancias.
    # Restamos cada valor de 'proxMat' de su máximo para que las distancias sean mayores cuando
    # las muestras caen juntas con menor frecuencia.
    distanceMat = proxMat.max() - proxMat
    
    return distanceMat


