# %%
import os
import sys
import pickle as pkl
import pandas as pd

from toolz import pipe
from rich import inspect

from typing import List

import matplotlib.pyplot as plt

# %%
src_path = os.path.join(os.getcwd(), "..")
if src_path not in sys.path:
    sys.path.append(src_path)

from src import Paths

# %%
paths = Paths(
    project=os.path.join(
        os.path.expanduser("~"), "Documents", "GitHub", "inequality-by-education"
    ),
    data=os.path.join(
        os.path.expanduser("~"),
        "Documents",
        "GitHub",
        "inequality-by-education",
        "data",
    ),
)
# %%
results = pkl.load(open(paths.join(paths.data, "results.pkl"), "rb"))
results = results.assign(n_exog=[len(x) for x in results.exog])


# %%
def get_relevant_instruments(results: pd.DataFrame) -> pd.DataFrame:
    relevant_exog = pipe(
        results,
        lambda x: x.query("f_stat > 10"),
        lambda x: x.query("f_pval <= 0.05"),
        lambda x: x.exog,
        lambda x: {i for i in x},
        lambda x: list(x),
        lambda x: sorted(x, key=lambda x: len(x)),
    )

    results["exog"] = results.exog.apply(lambda x: tuple(x))

    return results.query("exog in @relevant_exog")


relevant_results = get_relevant_instruments(results)
# %%
relevant_results = relevant_results.assign(
    coef=lambda x: [float(y.params.x) for y in x.iv]
).assign(coef_pval=lambda x: [float(y.pvalues.x) for y in x.iv])


# %%
def coef_vs_nexog(variable: str) -> None:
    pipe(
        relevant_results.query(f"dependent == '{variable}'"),
        lambda x: x.groupby("n_exog"),
        lambda x: [y[1].coef.values for y in x],
        lambda x: plt.boxplot(x),
        lambda x: plt.title(f"Coef per Number of Controls\n{variable}"),
        lambda x: plt.show(),
    )


coef_vs_nexog("Dln_wage_c")
coef_vs_nexog("Dunemp_rate_c")
coef_vs_nexog("Dnilf_rate_c")
