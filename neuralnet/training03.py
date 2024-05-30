from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import PredictionErrorDisplay
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import sys

sys.path.insert(1, "/home/angel/gitrepos/biosc/biosc")
from neuralnet import NeuralNetwork, Scaler, Scaler2

# Scaler pipeline
scaler01 = Scaler2()

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from tensorflow.keras.utils import plot_model
import evidential_deep_learning as edl

btsettl_df = pd.read_csv("../data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv")
btsettl_df.head()

# SE ESCOGEN LOS DATOS DE LAS COLUMNAS QUE VAMOS A USAR PARA EL ESTUDIO
selected_features = ["age_Myr", "M/Ms"]

X = btsettl_df[selected_features]
y = btsettl_df["Li"]
#
# ENTRENAMIENTO
y = y.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)


# IMPORTAMOS MinMaxScaler PARA ESCALAR LOS VALORES DE 'X' A VALORES COMPRENDIDOS ENTRE 0 Y 1
scaler = MinMaxScaler()
X_train_scaled0 = scaler.fit_transform(X_train)
X_test_scaled0 = scaler.fit_transform(X_test)

y_train_scaled0 = scaler.fit_transform(y_train)
y_test_scaled0 = scaler.fit_transform(y_test)

# sys.exit()


# def EvidentialRegressionLoss(true, pred):
#     return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)
#
#
# def build_and_compile_model():
#     model = Sequential(
#         [
#             Dense(units=60, activation="relu", input_shape=(2,)),
#             Dense(60, activation="relu"),
#             Dense(60, activation="relu"),
#             # Dense(1, activation="linear"),
#             edl.layers.DenseNormalGamma(1),
#         ]
#     )
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(1e-3),
#         loss=EvidentialRegressionLoss,
#     )
#     return model
#
#
# dnn_Li_model = build_and_compile_model()
#
# # # plot_model(dnn_Li_model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
#
# epoch_hist = dnn_Li_model.fit(
#     X_train, y_train, epochs=100, batch_size=50, validation_split=0.2
# )
#
# # AHORA HACE LA PREDICCION DE PRECIO
# param_pred = dnn_Li_model.predict(X_test)
#
# y_pred, v, alpha, beta = tf.split(param_pred, 4, axis=-1)
# y_pred = y_pred[:, 0]
# var = np.sqrt(beta / (v * (alpha - 1)))
# var = np.minimum(var, 1e3)[:, 0]  # for visualization
#
#
# fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
# PredictionErrorDisplay.from_predictions(
#     y_test.flatten(),
#     y_pred=y_pred,
#     kind="actual_vs_predicted",
#     # subsample=100,
#     ax=axs[0],
#     # random_state=0,
# )
# axs[0].set_title("Actual vs. Predicted values")
# PredictionErrorDisplay.from_predictions(
#     y_test.flatten(),
#     y_pred=y_pred,
#     kind="residual_vs_predicted",
#     # subsample=100,
#     ax=axs[1],
#     # random_state=0,
# )
# axs[1].set_title("Residuals vs. Predicted Values")
#
# fig.suptitle("Plotting cross-validated predictions")
# fig.savefig("edl_Li.png")
# plt.close()

# --------------------------------------
# Comparing with pre-trained NN in biosc
# --------------------------------------
age = X_test.values[:, 0]
mass = X_test.values[:, 1]

inputs = scaler01.transform(age, mass)

# Instantiate Pre-Trained Neural Network (BT-Settl)
nnet = NeuralNetwork()

# Neural Network predictions
y_pred, Pho = nnet.predict(inputs.T)

y_pred = y_pred.eval()

fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred=y_pred,
    kind="actual_vs_predicted",
    # subsample=100,
    ax=axs[0],
    # random_state=0,
)
axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred=y_pred,
    kind="residual_vs_predicted",
    # subsample=100,
    ax=axs[1],
    # random_state=0,
)
axs[1].set_title("Residuals vs. Predicted Values")

fig.suptitle("Plotting cross-validated predictions")
fig.savefig("NN_Li.png")
plt.close()


# plt.scatter(y_test, mu)
# plt.xlabel("Actual Li")
# plt.ylabel("Predicted Li")
# plt.title("Actual vs Predicted Li")
# plt.savefig("actual-predicted-Li.png")
# plt.close()
#
# residuals = y_test.flatten() - mu
# plt.scatter(mu, residuals)
# plt.xlabel("Predicted Li")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.axhline(y=0, color="r", linestyle="--")
# plt.savefig("residuals.png")
# plt.close()
#
#
# model = Sequential(
#     [
#         Dense(64, activation="relu", name="Age-Mass", input_dim=x_train.shape[1]),
#         Dense(1, activation="relu", name="Li"),
#         edl.layers.DenseNormalGamma(1),
#     ]
# )
#
#
# # plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
#
#
# def EvidentialRegressionLoss(true, pred):
#     return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)
#
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss=edl.losses.EvidentialRegression,  # Evidential loss!
# )
#
# model.fit(x_train, y_train, batch_size=10, epochs=10)
#
# y_pred = model(x_test)

# mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
# mu = mu[:, 0]
# var = np.sqrt(beta / (v * (alpha - 1)))
# var = np.minimum(var, 1e3)[:, 0]  # for visualization
#
# pd.DataFrame({"mu":mu,
#               "var":var}).plot.scatter(x="mu", y="var")

# print(mu)


# # # Initialize the KFold class
# # kfold = KFold(n_splits=5, shuffle=True)
#
# # # Train and evaluate the model using cross-validation
# # fold_loss_scores = []
# # for train, test in kfold.split(X, Y):
# #   # Fit the model on the training data
# #   model.fit(X[train], [Li[train], Pho[train]], epochs=300, batch_size=32, verbose=0)
#
# #   # Evaluate the model on the test data
# #   scores = model.evaluate(X[test], [Li[test], Pho[test]], verbose=0)
# #   fold_loss_scores.append(scores[0])
# #   print("loss: %.4f - Li_loss: %.4f - Photometry_loss: %.4f" % (scores[0], scores[1], scores[2]))
# # # Calculate the mean accuracy across all folds
# # mean_loss = np.mean(fold_loss_scores)
# # print("Mean loss: %.4f" % mean_loss)
