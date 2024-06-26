#!/apps/software/Anaconda3/2022.05/envs/tensorflow-gpu/bin/python

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from pickle import dump
from sklearn.metrics import PredictionErrorDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import evidential_deep_learning as edl

print(f"Running Tensoflow {tf.__version__}")

# Load data
BTSettl = pd.read_csv("../data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv")
inputs = ["age_Myr", "M/Ms"]
targets = [
    "Li",
    # "G",
    # "G_BP",
    # "G_RP",
    # "J",
    # "H",
    # "K",
    # "g_p1",
    # "r_p1",
    # "i_p1",
    # "y_p1",
    # "z_p1",
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

inputs = layers.Input(shape=(2,), name="Age_Mass")
dense = layers.Dense(
    units=64,
    activation="relu",
    kernel_initializer=keras.initializers.he_normal(),
    # kernel_regularizer = keras.regularizers.L2(0.01),
    bias_initializer=keras.initializers.Zeros(),
    name="HL1",
)(inputs)
Li = layers.Dense(
    units=1,
    activation="sigmoid",
    kernel_initializer=keras.initializers.GlorotNormal(),
    # use_bias = False,
    name="Li",
)(dense)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# # Define Huber loss
# huber = keras.losses.Huber()
# Define Mean Squared Error loss
# mse = keras.losses.MeanSquaredError()
# # Define Mean Absolute Error loss
mae = keras.losses.MeanAbsoluteError()
# # Define custom loss
# def custom_loss(y_true, y_pred):
#     """Increase the loss by a factor"""
#     alpha = 2.0
#     return alpha * mse(y_true, y_pred)
# Set metrics
# RMSE = keras.metrics.RootMeanSquaredError()

model0 = keras.Model(inputs=inputs, outputs=[Li], name="model0")
model0.compile(
    optimizer=optimizer,
    loss=mae,
)
# print(model.summary())

history = model0.fit(
    X_train,
    [Li_train],
    epochs=20000,
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

Li_pred = model.predict(X_test)

y_pred = Li_pred.flatten()

fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
PredictionErrorDisplay.from_predictions(
    Li_test.flatten(),
    y_pred=y_pred,
    kind="actual_vs_predicted",
    ax=axs[0],
)

axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    Li_test.flatten(),
    y_pred=y_pred,
    kind="residual_vs_predicted",
    ax=axs[1],
)
axs[1].set_title("Residuals vs. Predicted Values")

fig.savefig("Li.png")
plt.close()

# # Get optimal values
# weights = model.get_weights()

# Save optimal values
# dump(weights, open("weights_edl.pkl", "wb"))

# Save scalers
# dump(scalers, open("scalers_edl.pkl", "wb"))
