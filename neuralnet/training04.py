# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from pickle import dump
import sys
import evidential_deep_learning as edl

# import torch
# from alig.tf import AliG
from sklearn.metrics import PredictionErrorDisplay

print(f"Running Tensoflow {tf.__version__}")


# In[2]:


# Load data
BTSettl = pd.read_csv("../data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv")
inputs = ["age_Myr", "M/Ms"]
targets = [
    "Li",
    "G",
    "G_BP",
    "G_RP",
    "J",
    "H",
    "K",
    "g_p1",
    "r_p1",
    "i_p1",
    "y_p1",
    "z_p1",
]
X = np.array(BTSettl[inputs])
Y = np.array(BTSettl[targets])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Normalize data
BoxCox = PowerTransformer(method="box-cox", standardize=False).fit(X_train)
X_train = BoxCox.transform(X_train)
X_test = BoxCox.transform(X_test)
MinMax = MinMaxScaler().fit(X_train)
X_train = MinMax.transform(X_train)
X_test = MinMax.transform(X_test)
scalers = [BoxCox, MinMax]

# Split outputs
Li_train = Y_train[:, 0]
Li_test = Y_test[:, 0]
Pho_train = Y_train[:, 1:]
Pho_test = Y_test[:, 1:]
# # Define Neural Network: structure, activation, initializers, etc.

inputs = layers.Input(shape=(2,), name="Age_Mass")
dense = layers.Dense(
    units=64,
    activation="relu",
    kernel_initializer=keras.initializers.HeNormal(),
    # kernel_regularizer = keras.regularizers.L2(0.01),
    bias_initializer=keras.initializers.Zeros(),
    name="HL1",
)(inputs)
dense = layers.Dense(
    units=64,
    activation="relu",
    kernel_initializer=keras.initializers.HeNormal(),
    # kernel_regularizer = keras.regularizers.L2(0.01),
    bias_initializer=keras.initializers.Zeros(),
    name="HL2",
)(dense)
dense = layers.Dense(
    units=64,
    activation="relu",
    kernel_initializer=keras.initializers.HeNormal(),
    # kernel_regularizer = keras.regularizers.L2(0.01),
    bias_initializer=keras.initializers.Zeros(),
    name="HL3",
)(dense)
Li = edl.layers.DenseNormalGamma(1)(dense)  # Evidential distribution!
# Li = layers.Dense(
#     units=1,
#     activation="sigmoid",
#     kernel_initializer=keras.initializers.GlorotNormal(),
#     # use_bias = False,
#     name="Li",
# )(dense)
Photometry = edl.layers.DenseNormalGamma(11)(dense)  # Evidential distribution!
# Photometry = layers.Dense(
#     units=11,
#     # kernel_regularizer = keras.regularizers.L2(0.01),
#     # bias_regularizer = keras.regularizers.L2(0.01),
#     name="Photometry",
# )(dense)
#
#
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# # In[6]:

# # Define Huber loss
# huber = keras.losses.Huber()
#
# Define Mean Squared Error loss
mse = keras.losses.MeanSquaredError()

# # Define Mean Absolute Error loss
# msa = keras.losses.MeanAbsoluteError()
#
#
# # Define custom loss
# def custom_loss(y_true, y_pred):
#     """Increase the loss by a factor"""
#     alpha = 2.0
#     return alpha * mse(y_true, y_pred)
#
#
# # In[7]:
#
# edl_loss = edl.losses.EvidentialRegression  # Evidential loss!
# Custom loss function to handle the custom regularizer coefficient


def EvidentialRegressionLoss(true, pred):
    return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)


# Set metrics
RMSE = keras.metrics.RootMeanSquaredError()
#
#
# # In[8]:
#
#
# # Instanciate and compile model
model = keras.Model(inputs=inputs, outputs=[Li, Photometry], name="BTSettl")
model.compile(
    optimizer=optimizer,
    loss=EvidentialRegressionLoss,
)
#

# print(model.summary())

history = model.fit(
    X_train,
    [Li_train, Pho_train],
    epochs=200,
    batch_size=16,
    verbose=0,
)

loss = history.history["loss"]
iters = range(len(loss))


fig, axs = plt.subplots(1, 1, figsize=(15, 10))
fig.suptitle("Training loss")
axs.plot(iters, loss, label="Total loss")
fig.savefig("loss.png")
plt.close()

# sys.exit()
param_pred = model.predict(X_test)
y_pred, v, alpha, beta = tf.split(param_pred[0], 4, axis=-1)

# y_pred = y_pred.flatten()
var = np.sqrt(beta / (v * (alpha - 1)))
var = np.minimum(var, 1e3)[:, 0]  # for visualization

# print(Li_test.shape)
# print(y_pred.shape)
# print(var.shape)
# sys.exit()

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(
    Li_test,
    y_pred=y_pred[:, 0],
    kind="actual_vs_predicted",
    # subsample=100,
    ax=axs[0],
    # random_state=0,
)
axs[0].errorbar(y_pred[:, 0], Li_test, xerr=np.sqrt(var), fmt=".")

axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    Li_test,
    y_pred=y_pred[:, 0],
    kind="residual_vs_predicted",
    # subsample=100,
    ax=axs[1],
    # random_state=0,
)
axs[1].set_title("Residuals vs. Predicted Values")

# fig.suptitle("Plotting cross-validated predictions")
fig.savefig("edl_Li.png")
plt.close()

# # Get optimal values
weights = model.get_weights()

# Save optimal values
dump(weights, open("weights_edl.pkl", "wb"))

# Save scalers
dump(scalers, open("scalers_edl.pkl", "wb"))
