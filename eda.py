import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("T1.csv")

# Convert Date/Time
df["Date/Time"] = pd.to_datetime(
    df["Date/Time"],
    format="%d %m %Y %H:%M"
)

# -----------------------------
# Graph 1: Wind Speed vs Power
# -----------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Wind Speed (m/s)",
    y="LV ActivePower (kW)",
    data=df,
    alpha=0.3
)

plt.title("Wind Speed vs Power Generation")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Generation (kW)")
plt.show()


# ----------------------------------------
# Graph 2: Actual vs Theoretical Power
# ----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Theoretical_Power_Curve (KWh)"],
    df["LV ActivePower (kW)"],
    alpha=0.3
)

plt.title("Actual Power vs Theoretical Power")
plt.xlabel("Theoretical Power (KWh)")
plt.ylabel("Actual Power (kW)")
plt.show()


# --------------------------------
# Graph 3: Power Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["LV ActivePower (kW)"],
    bins=50,
    kde=True
)

plt.title("Power Generation Distribution")
plt.xlabel("Power Generation (kW)")
plt.ylabel("Frequency")
plt.show()