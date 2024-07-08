import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import seaborn as sns

with open("validation_Li_model_results.pkl", "rb") as f:
    results = pickle.load(f)

# Calculate average MSE for each model
average_results = {model: np.mean(errors) for model, errors in results.items()}
sd_results = {model: np.std(errors) for model, errors in results.items()}
best_model_name = min(average_results, key=average_results.get)
print(
    f"The best model is {best_model_name} with an average MSE of {average_results[best_model_name]:.4f}"
)
# df = pd.read_pickle("comparative.pkl")

summary_results = [average_results, sd_results]
df = pd.DataFrame(summary_results, index=list(range(len(summary_results)))).T
df.columns = ["Average", "StandardDeviation"]
x_values = range(len(df))

# Create the error bar plot
fig, axs = plt.subplots(ncols=1, figsize=(8, 8))
plt.errorbar(
    x_values,
    df["Average"],
    yerr=df["StandardDeviation"],
    fmt="o",
    capsize=5,
    capthick=2,
    ecolor="red",
    label="Average with Std Dev",
)

# Adding labels and title
plt.xlabel("Model")
plt.ylabel("Average")
plt.title("Error Bar Plot of Averages with Standard Deviations")
plt.legend()
fig.savefig("validation_Li_model_1.png")
plt.close()

# Probando
df = pd.DataFrame(results)  # poner todo en un data DataFrame

fig, axs = plt.subplots(ncols=1, figsize=(8, 8))
df.boxplot(ax=axs)
fig.savefig("validation_Li_model_2.png")
plt.close()
