import pandas as pd
import numpy as np


n_feat = 100
n_samples = 100

mz = np.random.uniform(0, 100, size=n_feat)
rt = np.random.uniform(0, 200, size=n_feat)
X = np.random.exponential(10, size=(n_feat, n_samples))

x_df = pd.DataFrame(
    np.concatenate([mz[:, None], rt[:, None], X], axis=1),
    index=[f"feature{i}" for i in range(n_feat)],
    columns=["mz", "rt"] + [f"sample{i}" for i in range(n_samples)],
)

group_df = pd.DataFrame(
    {
        "class": np.random.choice(["QC", "Subject"], size=n_samples),
        "injection.order": np.random.choice(n_samples, n_samples, replace=False),
        "batch": np.random.choice(4, n_samples, replace=True),
    },
    index=[f"sample{i}" for i in range(n_samples)],
)

x_df.to_csv("./example_x.csv")
group_df.to_csv("./example_sample_info.csv")
