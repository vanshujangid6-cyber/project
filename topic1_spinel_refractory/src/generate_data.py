"""
Generate a literature-informed synthetic MgO-Al2O3-SiO2 spinel-refractory
dataset for prototyping an ML composition/process-property pipeline.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 90


def generate_compositions(n=N_SAMPLES):
    mgo = RNG.uniform(4, 20, n)
    sio2 = RNG.uniform(0.3, 3.0, n)
    al2o3 = 100 - mgo - sio2
    return pd.DataFrame({
        "Al2O3_pct": al2o3, "MgO_pct": mgo, "SiO2_pct": sio2,
        "Sinter_Temp_C": RNG.uniform(1450, 1700, n),
        "Hold_Time_hr": RNG.uniform(2, 8, n),
    })


def compute_targets(df):
    mgo, sio2 = df["MgO_pct"].values, df["SiO2_pct"].values
    al2o3 = df["Al2O3_pct"].values
    T, t = df["Sinter_Temp_C"].values, df["Hold_Time_hr"].values

    T_kelvin = T + 273.15
    arrhenius = np.exp(-9500 / T_kelvin) * np.exp(9500 / (1600 + 273.15))
    expansion = 0.9 * (mgo / 15) ** 0.8 * arrhenius * (1 + 0.15 * np.sqrt(t / 5))
    expansion = np.clip(expansion + RNG.normal(0, 0.05, len(df)), 0.05, 3.5)

    densification = 1 / (1 + np.exp(-(T - 1550) / 40))
    porosity_penalty = (1 - densification) * (mgo / 20) * 2.5
    corrosion = 1.4 * sio2 - 0.03 * al2o3 + porosity_penalty + 0.6
    corrosion = np.clip(corrosion + RNG.normal(0, 0.25, len(df)), 0.2, 10.0)

    out = df.copy()
    out["Volumetric_Expansion_pct"] = np.round(expansion, 3)
    out["Slag_Corrosion_Index"] = np.round(corrosion, 3)
    return out


if __name__ == "__main__":
    data = compute_targets(generate_compositions())
    data.to_csv("/home/claude/topic1_spinel_refractory/data/spinel_data.csv", index=False)
    print(data.describe())
