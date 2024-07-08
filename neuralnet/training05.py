#!/apps/software/Anaconda3/2022.05/envs/tensorflow-gpu/bin/python
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from typing import List


from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from pickle import dump
# from sklearn.metrics import PredictionErrorDisplay

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sys

print(f"Running Tensoflow {tf.__version__}")


# Defining model archiquetures
# ----------------------------
def create_custom_model(
    input_shape: int,
    num_layers: int,
    units: List[int],
    activations: List[str],
    output_units: int,
    output_activation: str = "linear",
) -> Sequential:
    """
    Create a Keras Sequential model with the specified number of layers.

    Parameters:
    - input_shape (int): The number of features in the input data.
    - num_layers (int): The number of hidden layers in the model.
    - units (List[int]): A list containing the number of units for each hidden layer.
    - activations (List[str]): A list containing the activation function for each hidden layer.
    - output_units (int): The number of units in the output layer.
    - output_activation (str): The activation function for the output layer (default is 'linear').

    Returns:
    - model (Sequential): The compiled Keras Sequential model.
    """

    if num_layers != len(units) or num_layers != len(activations):
        raise ValueError(
            "The length of 'units' and 'activations' must be equal to 'num_layers'"
        )

    model = Sequential()

    # Input layer
    model.add(Dense(units[0], activation=activations[0], input_shape=(input_shape,)))

    # Hidden layers
    for i in range(1, num_layers):
        model.add(
            Dense(
                units[i],
                activation=activations[i],
                kernel_initializer=keras.initializers.HeNormal(),
                bias_initializer=keras.initializers.Zeros(),
            )
        )

    # Output layer
    model.add(
        Dense(
            output_units,
            activation=output_activation,
            kernel_initializer=keras.initializers.GlorotNormal(),
        )
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=keras.metrics.RootMeanSquaredError(),
    )

    return model


# Load data
# ---------

BTSettl = pd.read_csv("../data/BT-Settl_all_Myr_Gaia_2MASS_PanSTARRS_ALi.csv")
inputs = ["age_Myr", "M/Ms"]
targets = ["ALi"]

inputs = np.array(BTSettl[inputs])
targets = np.array(BTSettl[targets])

# Normalize data
BoxCox = PowerTransformer(method="box-cox", standardize=False).fit(inputs)
inputs = BoxCox.transform(inputs)
MinMax = MinMaxScaler().fit(inputs)
inputs = MinMax.transform(inputs)
scalers = [BoxCox, MinMax]


# Create the model
# ---------------

input_shape = 2  # Number of features in the input data
num_layers = 3  # Number of hidden layers
units = [64, 64, 64]  # Units in each hidden layer
activations = [
    # "relu",
    "relu",
    "relu",
    "relu",
]  # Activation functions for each hidden layer
output_units = 1  # Number of units in the output layer
output_activation = "relu"  # Activation function for the output layer

model = create_custom_model(
    input_shape, num_layers, units, activations, output_units, output_activation
)

# print(model.summary())

history = model.fit(
    inputs,
    targets,
    epochs=20000,
    batch_size=32,
    verbose=0,
)


# Plot loss vs. iter
loss = history.history["loss"]
iters = range(len(loss))

fig, axs = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle("Training vs. evaluation loss")
axs.set_title("Total loss")
axs.set_ylabel("log(msa)")
axs.plot(iters, loss, label="Total cost", zorder=1)
axs.set_yscale("log")
fig.savefig("loss.png")
plt.close()

# Plot RMSE vs. iter
Li_rmse = history.history["root_mean_squared_error"]
iters = range(len(Li_rmse))

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_title("RMSE")
ax.plot(iters, Li_rmse, label="Lithium")
ax.legend()
ax.set_ylabel("log(rmse)")
ax.set_yscale("log")
ax.legend()
fig.savefig("RMSE.png")
plt.close()

# Get optimal values
weights = model.get_weights()

# Save optimal values
dump(weights, open("weights_Ali_3x64.pkl", "wb"))

# Save scalers
dump(scalers, open("scalers_Ali_3x64.pkl", "wb"))

# Check weights correspondence
i = 0
for weight in weights:
    print(f"weights[{i}].shape = {weight.shape}")
    i += 1
#
