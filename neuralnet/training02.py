import functools
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
import evidential_deep_learning as edl

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from pickle import dump

print(f"Running Tensoflow {tf.__version__}")


# In[2]:


BTSettl = pd.read_csv("../data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv")
# Load data
# BTSettl = pd.read_csv('..\data\BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv')
inputs = ["age_Myr", "M/Ms"]
targets = [
    "Li",
]
X = np.array(BTSettl[inputs])
Y = np.array(BTSettl[targets])

pd.Series(Y.flatten()).describe()

# Split into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)


# plt.title("Inputs variables distribution")
# plt.boxplot(X_train, labels=["age_Myr", "M/Ms"])
# plt.show()
#
# plt.title("outputs variables distribution")
# plt.boxplot(Y_train)
# plt.show()

model = Sequential(
    [
        Input(shape=(2,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        edl.layers.DenseNormalGamma(1),
    ]
)


plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
#


def EvidentialRegressionLoss(true, pred):
    return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=edl.losses.EvidentialRegression,  # Evidential loss!
)

# model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=EvidentialRegressionLoss)

model.fit(X_train, Y_train, batch_size=10, epochs=500)

Y_pred = model(X_test)

mu, v, alpha, beta = tf.split(Y_pred, 4, axis=-1)
mu = mu[:, 0]
var = np.sqrt(beta / (v * (alpha - 1)))
var = np.minimum(var, 1e3)[:, 0]  # for visualization

print(mu)
print(var)

sys.exit()


def main():
    # Create some training and testing data
    x_train, y_train = my_data(-4, 4, 1000)
    x_test, y_test = my_data(-7, 7, 1000, train=False)

    # Define our model with an evidential output
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(2,), name="Age, Mass", activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(edl.layers.DenseNormalGamma(1))

    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)

    # Compile and fit the model!
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4), loss=EvidentialRegressionLoss
    )
    model.fit(x_train, y_train, batch_size=100, epochs=500)

    # Predict and plot using the trained model
    y_pred = model(x_test)
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)

    # Done!!


#### Helper functions ####
def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma).astype(np.float32)

    return x, y


def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1.0, c="#463c3c", zorder=0, label="Train")
    plt.plot(x_test, y_test, "r--", zorder=2, label="True")
    plt.plot(x_test, mu, color="#007cab", zorder=3, label="Pred")
    plt.plot([-4, -4], [-150, 150], "k--", alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], "k--", alpha=0.4, zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test,
            (mu - k * var),
            (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor="#00aeef",
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None,
        )
    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()


# Instanciate and compile model
model = keras.Model(inputs=inputs, outputs=[mu_Li, sigma_Li], name="BTSettl")
model = keras.Model(inputs=inputs, outputs=[Li, Photometry], name="BTSettl")
model.compile(optimizer=optimizer, loss={"Li": msa, "Photometry": mse}, metrics=[RMSE])

keras.utils.plot_model(
    model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
)
# In[9]:


model.summary()


# In[10]:


# Define early stopping feature from keras
# early_stop = keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 0.01, patience = 300, mode = 'min', restore_best_weights = True)


# In[11]:


history = model.fit(
    X_train,
    [Li_train, Pho_train],
    epochs=10000,
    batch_size=16,
    validation_data=(X_test, [Li_test, Pho_test]),
    verbose=0,
)
# , callbacks = [early_stop]


# In[12]:


# Plot loss vs. iter
loss = history.history["loss"]
val = history.history["val_loss"]
Li_loss = history.history["Li_loss"]
Li_val = history.history["val_Li_loss"]
Photometry_loss = history.history["Photometry_loss"]
Photometry_val = history.history["val_Photometry_loss"]
iters = range(len(loss))

fig, axs = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle("Training vs. evaluation loss")
axs[0].set_title("Total loss")
axs[0].set_ylabel("log(msa)")
axs[0].plot(iters, loss, label="Total cost", zorder=1)
axs[0].plot(iters, val, label="Total val cost", zorder=0)
axs[0].set_yscale("log")
axs[0].legend()
axs[1].set_title("Lithium loss")
axs[1].plot(iters, Li_loss, label="Training", zorder=1)
axs[1].plot(iters, Li_val, label="Validation", zorder=0)
axs[1].legend()
axs[1].set_yscale("log")
axs[1].set_ylabel("log(mse)")
axs[2].set_title("Photometry loss")
axs[2].plot(iters, Photometry_loss, label="Training", zorder=1)
axs[2].plot(iters, Photometry_val, label="Validation", zorder=0)
axs[2].set_xlabel("iter")
axs[2].set_ylabel("log(mse)")
axs[2].set_yscale("log")
axs[2].legend()
plt.show()


# In[13]:


# Plot RMSE vs. iter
Li_rmse = history.history["Li_root_mean_squared_error"]
Photometry_rmse = history.history["Photometry_root_mean_squared_error"]
iters = range(len(Li_rmse))

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.set_title("RMSE")
ax.plot(iters, Li_rmse, label="Lithium")
ax.legend()
ax.set_ylabel("log(rmse)")
ax.plot(iters, Photometry_rmse, label="Photometry")
ax.set_xlabel("iter")
ax.set_yscale("log")
ax.legend()
plt.show()


# In[14]:


# Evaluate model
model.evaluate(
    X_test, [Li_test, Pho_test], batch_size=16
)  # , callbacks = [early_stop])


# In[15]:


test_preds = model.predict(X_test)
train_preds = model.predict(X_train)


# In[16]:


# Model evaluation
o = 0.5  # Opacity
# Lithium
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle("Test Dataset")
axs[0].scatter(X_test[:, 0], Li_test, alpha=o, label="True values")
axs[0].scatter(X_test[:, 0], test_preds[0], alpha=o, label="Predictions")
axs[0].set_xlabel("Age")
axs[0].set_ylabel(targets[0])
axs[0].legend()
axs[1].scatter(X_test[:, 1], Li_test, alpha=o, label="True values")
axs[1].scatter(X_test[:, 1], test_preds[0][:, 0], alpha=o, label="Predictions")
axs[1].set_xlabel("Mass")
axs[1].legend()
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle("Train Dataset")
axs[0].scatter(X_train[:, 0], Li_train, alpha=o, label="True values")
axs[0].scatter(X_train[:, 0], train_preds[0], alpha=o, label="Predictions")
axs[0].set_xlabel("Age")
axs[0].set_ylabel(targets[0])
axs[0].legend()
axs[1].scatter(X_train[:, 1], Li_train, alpha=o, label="True values")
axs[1].scatter(X_train[:, 1], train_preds[0][:, 0], alpha=o, label="Predictions")
axs[1].set_xlabel("Mass")
axs[1].legend()
plt.show()


# In[17]:


for i in range(test_preds[1].shape[1]):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
    fig.suptitle("Test dataset")
    axs[0].scatter(X_test[:, 0], Pho_test[:, i], alpha=o, label="True values")
    axs[0].scatter(X_test[:, 0], test_preds[1][:, i], alpha=o, label="Predictions")
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel(targets[i + 1])
    axs[0].legend()
    axs[1].scatter(X_test[:, 1], Pho_test[:, i], alpha=o, label="True values")
    axs[1].scatter(X_test[:, 1], test_preds[1][:, i], alpha=o, label="Predictions")
    axs[1].set_xlabel("Mass")
    axs[1].legend()
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
    fig.suptitle("Train dataset")
    axs[0].scatter(X_train[:, 0], Pho_train[:, i], alpha=o, label="True values")
    axs[0].scatter(X_train[:, 0], train_preds[1][:, i], alpha=o, label="Predictions")
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel(targets[i + 1])
    axs[0].legend()
    axs[1].scatter(X_train[:, 1], Pho_train[:, i], alpha=o, label="True values")
    axs[1].scatter(X_train[:, 1], train_preds[1][:, i], alpha=o, label="Predictions")
    axs[1].set_xlabel("Mass")
    axs[1].legend()
    plt.show()


# In[22]:


# Get optimal values
weights = model.get_weights()

# Save optimal values
dump(weights, open("weights.pkl", "wb"))


# In[23]:


# Save scalers
dump(scalers, open("scalers.pkl", "wb"))


# In[20]:


# Check weights correspondence
i = 0
for weight in weights:
    print(f"weights[{i}].shape = {weight.shape}")
    i += 1


# In[21]:


# # Normalize data
# scaler = StandardScaler().fit(X)
# X = scaler.transform(X)

# # Split outputs data
# Li = Y[:, 0]
# Pho = Y[:, 1:]

# # Initialize the KFold class
# kfold = KFold(n_splits=5, shuffle=True)

# # Train and evaluate the model using cross-validation
# fold_loss_scores = []
# for train, test in kfold.split(X, Y):
#   # Fit the model on the training data
#   model.fit(X[train], [Li[train], Pho[train]], epochs=300, batch_size=32, verbose=0)

#   # Evaluate the model on the test data
#   scores = model.evaluate(X[test], [Li[test], Pho[test]], verbose=0)
#   fold_loss_scores.append(scores[0])
#   print("loss: %.4f - Li_loss: %.4f - Photometry_loss: %.4f" % (scores[0], scores[1], scores[2]))
# # Calculate the mean accuracy across all folds
# mean_loss = np.mean(fold_loss_scores)
# print("Mean loss: %.4f" % mean_loss)
