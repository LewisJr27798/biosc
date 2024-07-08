from sklearn.model_selection import RepeatedKFold, train_test_split, KFold
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from typing import List
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import PredictionErrorDisplay

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
import sys
import os
import pickle

print(f"Running Tensoflow {tf.__version__}")

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


# Create the model 2
input_shape = 2  # Number of features in the input data
num_layers = 2  # Number of hidden layers
units = [64, 64]  # Units in each hidden layer
activations = ["relu", "relu"]  # Activation functions for each hidden layer
output_units = 1  # Number of units in the output layer
output_activation = "relu"  # Activation function for the output layer

model1 = create_custom_model(
    input_shape, num_layers, units, activations, output_units, output_activation
)

# Create the model 3
input_shape = 2  # Number of features in the input data
num_layers = 3  # Number of hidden layers
units = [64, 64, 64]  # Units in each hidden layer
activations = ["relu", "relu", "relu"]  # Activation functions for each hidden layer
output_units = 1  # Number of units in the output layer
output_activation = "relu"  # Activation function for the output layer

model2 = create_custom_model(
    input_shape, num_layers, units, activations, output_units, output_activation
)

# Create the model 1
input_shape = 2  # Number of features in the input data
num_layers = 4  # Number of hidden layers
units = [64, 64, 64, 64]  # Units in each hidden layer
activations = [
    "relu",
    "relu",
    "relu",
    "relu",
]  # Activation functions for each hidden layer
output_units = 1  # Number of units in the output layer
output_activation = "relu"  # Activation function for the output layer

model3 = create_custom_model(
    input_shape, num_layers, units, activations, output_units, output_activation
)

# Create the model 1
input_shape = 2  # Number of features in the input data
num_layers = 4  # Number of hidden layers
units = [32, 32, 32, 32]  # Units in each hidden layer
activations = [
    "relu",
    "relu",
    "relu",
    "relu",
]  # Activation functions for each hidden layer
output_units = 1  # Number of units in the output layer
output_activation = "relu"  # Activation function for the output layer

model4 = create_custom_model(
    input_shape, num_layers, units, activations, output_units, output_activation
)


models = [model1, model2, model3, model4]

# Directory to save models
os.makedirs("best_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

kfold = RepeatedKFold(n_repeats=5, random_state=72)

results = {}

for model in models:
    model_name = model.name
    fold_errors = []
    for fold, (train_index, val_index) in enumerate(kfold.split(inputs, targets)):
        X_train, X_val = inputs[train_index], inputs[val_index]
        y_train, y_val = targets[train_index], targets[val_index]

        # Save the best model for each fold
        # csv_logger = CSVLogger(
        #     f"logs/temp_trainin_log_{model_name}_fold_{fold}.csv", append=False
        # )

        checkpoint = ModelCheckpoint(
            f"best_models/model_{model_name}_fold_{fold}.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=20000,
            batch_size=32,
            verbose=0,
            validation_data=(X_val, y_val),
            # callbacks=[csv_logger, checkpoint],
            callbacks=[checkpoint],
        )

        # Load the best model for the current fold
        model.load_weights(f"best_models/model_{model_name}_fold_{fold}.h5")
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        fold_errors.append(mse)

        # fig, axs = plt.subplots(1, 1, figsize=(15, 10))
        # fig.suptitle("Training loss")
        # axs.plot(
        #     range(len(history.history["loss"])),
        #     history.history["loss"],
        #     label="Total loss",
        # )
        # fig.savefig(f"logs/loss_{model_name}_fold_{fold}.png")
        # plt.close()

    results[model_name] = fold_errors


with open("validation_Li_model_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Calculate average MSE for each model
# average_results = {model: np.mean(errors) for model, errors in results.items()}
# sd_results = {model: np.std(errors) for model, errors in results.items()}
# best_model_name = min(average_results, key=average_results.get)
# print(
#     f"The best model is {best_model_name} with an average MSE of {average_results[best_model_name]:.4f}"
# )
#
# results = [average_results, sd_results]
# df = pd.DataFrame(results, index=list(range(len(results)))).T
# df.columns = ["Average", "StandardDeviation"]
#

# x_values = range(len(df))
#
# # Create the error bar plot
# fig, axs = plt.subplots(ncols=1, figsize=(8, 8))
# plt.errorbar(
#     x_values,
#     df["Average"],
#     yerr=df["StandardDeviation"],
#     fmt="o",
#     capsize=5,
#     capthick=2,
#     ecolor="red",
#     label="Average with Std Dev",
# )
#
# # Adding labels and title
# plt.xlabel("Model")
# plt.ylabel("Average")
# plt.title("Error Bar Plot of Averages with Standard Deviations")
# plt.legend()
#
# fig.savefig("comparative.png")
#
# # Save the best model overall (based on average MSE across folds)
# best_model_overall = (
#     create_model_1() if best_model_name == "create_model_1" else create_model_2()
# )
# best_model_overall.load_weights(
#     f"best_models/{best_model_name}_fold_{np.argmin(results[best_model_name])}.h5"
# )
# best_model_overall.save("best_models/best_model_overall.h5")
#

# def plot_figures(history, model):
#     loss = history.history["loss"]
#     iters = range(len(loss))
#
#     fig, axs = plt.subplots(1, 1, figsize=(15, 10))
#     fig.suptitle("Training loss")
#     axs.plot(iters, loss, label="Total loss")
#     fig.savefig(f"{model.name}_loss.png", format="png")
#     plt.close()
#
#     Li_pred = model.predict(X_test)
#
#     y_pred = Li_pred.flatten()
#
#     fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
#     PredictionErrorDisplay.from_predictions(
#         Li_test.flatten(),
#         y_pred=y_pred,
#         kind="actual_vs_predicted",
#         ax=axs[0],
#     )
#
#     axs[0].set_title("Actual vs. Predicted values")
#     PredictionErrorDisplay.from_predictions(
#         Li_test.flatten(),
#         y_pred=y_pred,
#         kind="residual_vs_predicted",
#         ax=axs[1],
#     )
#     axs[1].set_title("Residuals vs. Predicted Values")
#     fig.savefig(f"{model.name}_Li.png", format="png")
#     plt.close()
#

#
# Define the K-fold Cross Validator
# num_folds = 10
# kfold = KFold(n_splits=num_folds, shuffle=True)
#
# # K-fold Cross Validation model evaluation
# batch_size = 16
# no_epochs = 10
# verbosity = 0
# t0 = time.time()
# fold_no = 1
# for train, test in kfold.split(inputs, targets):
#     # Generate a print
#     print("------------------------------------------------------------------------")
#     print(f"Training for fold {fold_no} ...")
#     history0 = model0.fit(
#         inputs[train],
#         targets[train],
#         batch_size=batch_size,
#         epochs=no_epochs,
#         verbose=verbosity,
#     )
#     history1 = model1.fit(
#         inputs[train],
#         targets[train],
#         batch_size=batch_size,
#         epochs=no_epochs,
#         verbose=verbosity,
#     )
#     history2 = model2.fit(
#         inputs[train],
#         targets[train],
#         batch_size=batch_size,
#         epochs=no_epochs,
#         verbose=verbosity,
#     )
#     if fold_no == 1:
#         plot_figures(history0, model0)
#         plot_figures(history1, model1)
#         plot_figures(history2, model2)
#
#     for model in models:
#         scores = model.evaluate(inputs[test], targets[test], verbose=0)
#         print(
#             f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}"
#         )
#     # Increase fold number
#     fold_no = fold_no + 1
#
# t1 = time.time()
# total = t1 - t0
# print(total)
