import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

btsettl_df = pd.read_csv("../data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv")
btsettl_df.head()

# SE ESCOGEN LOS DATOS DE LAS COLUMNAS QUE VAMOS A USAR PARA EL ESTUDIO
selected_features = ["age_Myr", "M/Ms"]

X = btsettl_df[selected_features]
y = btsettl_df["Li"]

# IMPORTAMOS MinMaxScaler PARA ESCALAR LOS VALORES DE 'X' A VALORES COMPRENDIDOS ENTRE 0 Y 1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# AHORA NORMALIZAMOS EL OUTPUT Y LO ESCALAMOS IGUAL QUE 'X'
y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)

# ENTRENAMIENTO
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)

# DEFINIR MODELO
model = tf.keras.models.Sequential()

# SE DEFINEN 100 NEURONAS CON units, Y EL input_shape QUE SON LAS ENTRADAS, EN ESTE CASO 'X' QUE SON LAS selected_features
model.add(tf.keras.layers.Dense(units=60, activation="relu", input_shape=(2,)))
# AÑADIMOS DOS CAPAS MAS

model.add(tf.keras.layers.Dense(units=60, activation="relu"))
model.add(tf.keras.layers.Dense(units=60, activation="relu"))

# AHORA SE AÑADE LA NEURONA DE SALIDA
model.add(tf.keras.layers.Dense(units=1, activation="linear"))

# CON model.summary() VEMOS LA ESTRUCTURA DE LA RED
model.summary()

# AHORA HAY QUE COMPILAR
model.compile(optimizer="Adam", loss="mean_squared_error")

# Y AHORA SE ENTRENA EL MODELO
epoch_hist = model.fit(
    X_train, y_train, epochs=100, batch_size=50, validation_split=0.2
)

# AHORA HACE LA PREDICCION DE PRECIO
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Li")
plt.ylabel("Predicted Li")
plt.title("Actual vs Predicted Li")
plt.savefig("actual-predicted-Li.png")
plt.close()

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Li")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color="r", linestyle="--")
plt.savefig("residuals.png")
plt.close()
