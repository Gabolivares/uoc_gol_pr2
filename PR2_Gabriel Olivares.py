#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("max abs chile - max_abs_temp_chile.csv")


# In[3]:


df = df.rename(columns={'Unnamed: 0': 'año_mes'})


# In[4]:


df.head()


# In[5]:


# Las variables que explique antes
df.columns


# In[6]:


# el dataset contiene 28 variables y 499 registros

df.shape


# In[7]:


print(df.dtypes)


# Podemos ver que todos los datos son float, lo cual es correcto, ya que las temperaturas son decimales. A excepcion del primero el cual es un objeto, ya que tiene una cadena de texto

# Al realizar el PR1, el detaset original contenia datos como .. cuando estos eran missing values, estos se han cambiado por 99.9, por lo que se debiesen contabilizar como valores NAN.
# Por otro lado, los NAN existentes en la base de datos, a diferencia de 99.9 son para cuando la estacion no estaba activa aun.
# 

# In[8]:


df = df.replace(99.9, np.nan)


# In[9]:


df.describe()


# Se realiza un describe() para observar la distribucion de los datos y ver si hay algun dato extraño.
# 

# In[10]:


#calcularemos cuantos missing values son:
def contar_na(dataframe):
    for x in dataframe.columns:
        na_count = dataframe[x].isna().sum()
        print(f"{x}: ", na_count)


# In[11]:


contar_na(df)


# Podemos ver que la mayor cantidad de missing values pertenecen a la estacion Quinteros, El Loa, Graneros, Vallenar, Jardin Botanico, Chamonate, Cerrillos.
# 
# Dado que la cantidad de datos son 499 por estacion, podemos ver que ninguna de estas estaciones son utiles para evaluar, ya que la cantidad de missing values es mas de la mitad, a excepcion de Cerrillos y  Chamote. Sin embargo de igual forma se quitan ya que realizar una sustitucion por imputación nublaria las conclusiones.

# In[12]:


comunas_a_eliminar = ['Quintero', 'El Loa', 'Graneros', 'Vallenar', 'Jardín Botánico', 'Chamonate', 'Cerrillos']


# In[13]:


df = df.drop(columns=comunas_a_eliminar)


# In[14]:


df.columns


# Ahora tenemos missing values en las siguientes estaciones:

# In[15]:


contar_na(df)


#  31. Los missing values, los cambiaremos por la media de las temperaturas observadas en el mismo periodo.

# In[16]:


# funcion para reemplazar NaN por el promedio
def replace_nan_mean(fila):
    row_mean = fila.mean()
    return fila.fillna(row_mean)  # Rellenar NaN con el promedio



# In[17]:


column_means = df.iloc[:, 1:].mean()
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.fillna(col.mean()))


# In[18]:


df_fix = df.copy()


# In[19]:


contar_na(df_fix)


# In[20]:


df_fix.head()


# In[21]:


# df_fix['añoMes'] = df_fix.index



# In[22]:


df_fix[['año', 'mes']] = df_fix['año_mes'].str.split(expand=True)


# In[23]:


df_fix['mes'] = df_fix['mes'].str.lower()


# In[24]:


meses_a_numeros = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}

# Reemplazar mes por numero categorico, no se utiliza factor.
df_fix['mes'] = df_fix['mes'].map(meses_a_numeros)


# In[25]:


df_fix


# In[26]:


df = df_fix.iloc[:, 1:-2].copy()


# In[27]:


from scipy.stats import zscore
import matplotlib.pyplot as plt


for estacion in df.columns:
    print(f"Estación: {estacion}")
    print(df[estacion].describe())

    # Grafico

    plt.figure(figsize=(10, 6))
    plt.hist(df[estacion], bins=20, color='lightblue', edgecolor='black')
    plt.xlabel('Temperatura Maxima')
    plt.ylabel('Frec')
    plt.title(f'Distribución de Temperaturas Maximas - Estación {estacion}')
    plt.show()

    # aplica zscore para encontrar outliners

    z_scores = zscore(df[estacion])
    threshold = 3
    outliers = df[estacion][abs(z_scores) > threshold]
    print("outliners:")
    print(outliers)


# In[28]:


df_noLabel = df_fix.iloc[:, :-3]


# In[29]:


# podria quitar la estacion de la antartica? para mantenernos en el continente?


# In[30]:


for estacion in df.columns:
    plt.figure(figsize=(8, 6))
    boxplot = plt.boxplot(df[estacion])
    plt.title(f'Boxplot de Temperatura Máxima {estacion}')
    plt.ylabel('Temperatura (ºC)')
    plt.show()
    
    whiskers = [item.get_ydata() for item in boxplot['whiskers']]

    outliers = df[estacion][df[estacion] > whiskers[1][1]]
    print(f"outliners en la estación {estacion}:")
    print(outliers)


# Si bien un par de estaciones meteorologicas tienen temperaturas consideradas como outliners, podemos ver que estas temperaturas todas por encima del limite superior del grafico, se pueden cosiderar como correctas, ya que no son temperaturas absurdas. Además, al estar sobre el limite superior indica que la información de los gráficos está bien ya que al solo tener temperaturas máximas muestra los "outliners" con temperaturas altas.

# 34. ya que el dataset ha tenido una primera fase de limpieza en el momento de extraccion, solo ha sido necesario limpiar los NA de la base de datos.
# Esto fue lo que hicimos en el paso anterior.

# In[31]:


# 4.1 Aplica un modelo supervisado y uno no supervisado a los datos


# In[32]:


# NO SUPERVISADO SIN ESCALAR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# In[33]:


df_sample = df.sample(frac=0.8, random_state=42)
df_sample.head()


# In[34]:


correlation_matrix = df_sample.corr()

# sns.set(style="white")

plt.figure(figsize=(20,16))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de Correlacion entre estaciones")
plt.show


# Con esta matriz de correlaicon podemos ver que todas las estaciones tienen una correlacion positiva, es decir, a medida que aumenta la temperatura en una, aumenta la otra.
# Esto explica que el calentamiento global ha ido en aumento, ya que todas las variables han sido cada vez mayor

# In[ ]:





# In[35]:


# PCA para dejar el dataset en dos variables y ver la agrupacion del comportamiento del pais en general con respecto a los meses.
from sklearn.decomposition import PCA


# In[36]:


pca = PCA(n_components=2)

# ajustar el modelo PCA a los datos
pca.fit(df_sample)

# transformar datos usando el modelo PCA
X_pca = pca.transform(df_sample)


# In[ ]:





# In[37]:


componentes_principales = pca.components_
componentes_principales


# In[38]:


df_pca = pd.DataFrame(X_pca)


# In[39]:


k_range = range(1, 11)

sse = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_sample)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Número de Clusters, k')
plt.ylabel('Suma de los Errores Cuadráticos')
plt.title('Método del Codo (Sin Escalar)')
plt.grid(True)
plt.show()


# In[40]:


sse


# Con el metodo del codo, podemos ver que los K_neighbourgh y la sse podrían tomar valores como 3 o 4. Esto tiene sentido, ya que podemos etiquetar los datos, por estacion del año, ya sea verano, otoño, invierno, primavera.

# In[41]:


df_copy = df_fix.copy()


# In[42]:


df_copy = df_copy.drop(columns=["año_mes"])


# In[43]:


df_copy = pd.merge(df_sample, df_copy, how='left')


# In[44]:


def clustering_func(clusters_number, dataframe, column):
    
    kmeans = KMeans(n_clusters=clusters_number, random_state=42)

    # Entrenar 
    kmeans.fit(dataframe)

    # Predecir 
    clusters = kmeans.predict(dataframe)
    
    colors = plt.cm.viridis(np.linspace(0, 1, clusters_number))


    # Visualizar 
    plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], c=df_copy[column], cmap='viridis', alpha=0.5)  # Visualizar puntos de datos con colores de cluster
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='red', s=200, label='Centroids')  # Visualizar centroides en rojo
    plt.xlabel(dataframe.columns[0])
    plt.ylabel(dataframe.columns[1])
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()
    return kmeans


# In[45]:


kmeans = clustering_func(4,df_pca, 'mes')


# Lo que tengo aca es el grafico de los componentes principales, los cuales resume las temperaturas mensuales por año de cada estadion de medición.
# Tenemos que los puntos corresponden a los meses en los que estan tomadas la medicion.
# Con el fin de agrupar estos datos y verificar a que estacion del año corresponden. 
# Es decir, se ven que los mesese de enero y diciembre se agrupan en un grupo especifico, lo cual podemos llamarlo meses de altas temperaturas.
# Luego los meses mas calipsos se agrupan en conjunto y vemos que podrían ser meses más frios, los cuales Junio, Julio, quizas mayo.
# Finalemnte, podemos ver que los puntos verdosos son los meses como primavera y otoño, ya que estan juntos y sus temperaturas debiesen ser similares, unas mas altas que otras, pero en general se comportan similar.

# In[46]:


df_copy['Cluster'] = kmeans.labels_


# In[47]:


df_copy


# In[48]:


# modelo supervisado.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[49]:


df_copy2 = df_copy.copy()


# In[50]:


columnas_excluidas = df_copy2.columns[-3:]
columnas_a_promediar = df_copy2.columns.difference(columnas_excluidas)

# calcular el promedio de cada fila sin incluir las ultimas tres columnas
df_copy2['mean'] = df_copy2[columnas_a_promediar].mean(axis=1)


# In[51]:


df_copy2


# In[52]:


# Agrupamos por año y calculamos la temperatura promedio anual
df_yearly = df_copy2.groupby('año')['mean'].mean().reset_index()
df_yearly.head()


# In[53]:


# variables independientes (X) y dependientes (y)
X = df_yearly['año']
y = df_yearly['mean']


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[55]:


X = df_yearly[['año']].values.reshape(-1, 1)

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)



# In[56]:


X = [year[0] for year in X]


# In[57]:


plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel("Año")
plt.ylabel("Temperatura Promedio")
plt.title("Tendencia de las Temperaturas con los Años")
plt.xticks(rotation='vertical')
plt.show()


# Bajo este modelo y grafico, podemos ver que tomando en cuenta el promedio de las temperaturas anuales de todas las estaciones de medicion. Las temperaturas maximas absolutas estan en aumento, por ende. En Chile entre los ultimos 30 años ha habido un aumento en el promedio del maximo de 1º C.
# 

# In[58]:


print(df_yearly.iloc[:5]['mean'].mean())
print(df_yearly.iloc[-5:]['mean'].mean())


# In[59]:


from scipy.stats import kstest
from scipy.stats import mannwhitneyu


# In[61]:


# 4.2

# Primero que todo veo si estan con una distribucion normal con el metodo Kolmogorov-Smirnov

test_statistic, p_value = kstest(df_yearly['mean'], 'norm')
print(f"Estadistico de prueba: {test_statistic}")
print(f"p-value: {p_value}")

alpha = 0.05
if p_value > alpha:
    print("Parecen seguir una distribución normal.")
else:
    print("No siguen una distribución normal.")


# ya que la distribución de los datos no es normal, se utiliza el metodo Mann-Whitney U para comparar dos set de datos

# In[62]:


#tomando encuenta el promedio de los ultimos 5 años y primeros 5 años.
statistic, p_value = mannwhitneyu(df_yearly.iloc[:5]['mean'], df_yearly.iloc[-5:]['mean'])
print(f"Estadistica de prueba: {statistic}")
print(f"p-value: {p_value}")
# Interpretación del resultado
alpha = 0.05
if p_value < alpha:
    print("Se rechaza la hipótesis nula")
else:
    print("No hay evidencia significativa para rechazar la hipótesis nula")


# Como se rechaza la hipótesis nula, esto significa que hay evidencia significativa para creer en que existe una diferencia entre los promedios de los primeros 5 años y los últimos 5 años en cuanto a las temperaturas.
# Afirmando así lo que hemos dicho con la regresion

# In[ ]:




