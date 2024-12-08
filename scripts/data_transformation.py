# %% [markdown]
# # Assignment 3: Impact of low-skill immigration on the skill premium
# #### By: Augusto Ospital. May 17, 2023, Updated version: November 5th, 2023 (Mallika Chandra)

# %% [markdown]
# Data: 1990 and 2000 Census, and CONSPUMA as a spatial unit.
#
# __Step I__. Construct native labor-market outcomes by CONSPUMA $c$, year $y$ (1990 and 2000), and education $e$ ($e = H$ for ≥ 12 years schooling vs. $L$ for < 12 years)
#
# __Step II__. Construct working-age population with ($e = H$) and without ($e = L$) more than 12 years of schooling by $cy$, $N^e_{c,y}$
#
# __Step III__. Measure the change in the share with ≤ 12 years of schooling:
# $$ x_c = \frac{N^L_{c,2000}}{N^L_{c,2000}+N^H_{c,2000}} - \frac{N^L_{c,1990}}{N^L_{c,1990}+N^H_{c,1990}}$$
#
# __Step IV__. Construct instrument:
# $$ z_c
# = \frac{(I^{Mex}_{c,1990}/I^{Mex}_{1990})(I^{Mex}_{c,2000}-I^{Mex}_{c,1990})}{N^{L}_{c,1990}+N^{H}_{c,1990}}
# = \frac{I^{Mex}_{c,1990}}{N^{L}_{c,1990}+N^{H}_{c,1990}} \frac{I^{Mex}_{c,2000}-I^{Mex}_{c,1990}}{I^{Mex}_{1990}}
# $$
# where
# - $I^{Mex}_{c,y} = $ mexican population of $c$ in year $y$
# - $I^{Mex}_{y} = $ total mexican population in year $y$
#
# __Step V__. Using 2SLS, project changes in CONSPUMA relative outcomes for higher vs. lower education on $x_c$, instrumenting with $z_c$

# %% [markdown]
# ## Code Preliminaries

# %%
from pathlib import Path
import pandas as pd
from econtools import group_id
import numpy as np

import os
from dataclasses import dataclass
from rich import inspect
from toolz import pipe

# %%
# For regressions
import statsmodels.api as sm
from stargazer.stargazer import Stargazer  # nice tables with statsmodels
from linearmodels.iv import IV2SLS, compare  # 2sls with clustered SEs
import matplotlib.pyplot as plt

import pickle


# %%
@dataclass(frozen=True)
class Paths:
    project: str
    data: str
    home: str = os.path.expanduser("~")

    def join(self, *args):
        return os.path.join(*args)


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
# %% [markdown]
# ## Prepare the Data

# %% [markdown]
# #### Load the data from IPUMS

# %%
# Take a look at column definitions:

pd.read_stata(paths.join(paths.data, "usa_00122.dta"), iterator=True).variable_labels()
# %%
df = pd.read_stata(paths.join(paths.data, "usa_00122.dta"), convert_categoricals=False)
df = df[(df.age >= 18) & (df.age <= 65) & (df.gq <= 2)]
# For precision in sums:
df = df.astype(np.float64)

df["is_1990"] = np.where(df.year == 1990, 1, 0)
df["is_2000"] = np.where(df.year == 2000, 1, 0)

# Demographics:
df["is_native"] = np.where(df.bpl <= 120, 1, 0)
df["is_foreign"] = np.where((df.bpl > 120) & (df.bpl < 900), 1, 0)
df["is_female"] = np.where(df.sex == 2, 1, 0)
df["is_mex"] = np.where(df.bpl == 200, 1, 0)  # mexican dummy

# Education:
df["is_col"] = np.where(df.educ >= 10, 1, 0)
df["is_hs"] = np.where(df.educ >= 6, 1, 0)
df["is_low"] = np.where(df.educ < 6, 1, 0)  # low education dummy (less than HS)

# Employment status:
df["is_emp"] = np.where(df.empstat == 1, 1, 0)
df["is_unemp"] = np.where(df.empstat == 2, 1, 0)
df["is_nilf"] = np.where(df.empstat == 3, 1, 0)

# Manufacturing employment:
df["is_manuf"] = np.where(
    (df.is_emp == 1) & (df.ind1990 >= 100) & (df.ind1990 < 400), 1, 0
)
df["is_nonmanuf"] = np.where(
    (df.is_emp == 1) & ((df.ind1990 < 100) | (df.ind1990 >= 400)), 1, 0
)

# for numerator of instrument
df["mex_y"] = df.is_mex * df.perwt
# number of Mexicans per year
df["mex_y"] = df.groupby("year")["mex_y"].transform(sum)

# c-y controls and weight
df["mex_cy"] = df.is_mex * df.perwt
df["pop_cy"] = df.perwt
df["manuf_cy"] = df.is_manuf * df.is_emp * df.perwt
df["female_cy"] = df.is_female * df.is_emp * df.perwt
df["emp_cy"] = df.is_emp * df.perwt
df["col_cy"] = df.is_col * df.perwt
df["hs_cy"] = df.is_hs * df.perwt
df["fborn_cy"] = df.is_foreign * df.perwt
df["nborn_cy"] = df.is_native * df.perwt

for col in [c for c in df.columns if "_cy" in c]:
    df[col] = df.groupby(["year", "conspuma"])[col].transform(sum)

# manufacturing share of employed
df["manuf_share_cy"] = df["manuf_cy"] / df["emp_cy"]
# female share of employed
df["female_share_cy"] = df["female_cy"] / df["emp_cy"]
# college share of population
df["col_share_cy"] = df["col_cy"] / df["pop_cy"]
# high school share of population
df["hs_share_cy"] = df["hs_cy"] / df["pop_cy"]
# log of population (in age range)
df["lnpop_cy"] = np.log(df["pop_cy"])
df.loc[df["lnpop_cy"] == -np.inf, "lnpop_cy"] = np.nan
# foreign-born share of employed
df["fborn_share_cy"] = df["fborn_cy"] / (df["fborn_cy"] + df["nborn_cy"])
# Mexican share of population
df["mex_share_cy"] = df["mex_cy"] / df["pop_cy"]

cy_cols = [
    "mex_cy",
    "manuf_share_cy",
    "female_share_cy",
    "col_share_cy",
    "hs_share_cy",
    "lnpop_cy",
    "fborn_share_cy",
    "mex_share_cy",
]

# c-e-y outcomes for natives by conspuma, education, year
df["hours"] = df["uhrswork"] * df["wkswork1"]
df["pop_cey"] = df.groupby(["conspuma", "is_low", "year"])["perwt"].transform(sum)

# excluding US OUTLYING AREAS/TERRITORIES
df = df[df["bpl"] < 100].copy()

df["nilf_cey"] = df["is_nilf"] * df["perwt"]
df["unemp_cey"] = df["is_unemp"] * df["perwt"]
df["emp_cey"] = df["is_emp"] * df["perwt"]
df["inc_cey"] = df["incwage"] * df["perwt"]
df["hours_cey"] = df["hours"] * df["perwt"]
for col in ["nilf_cey", "unemp_cey", "emp_cey", "inc_cey", "hours_cey"]:
    df[col] = df.groupby(["year", "conspuma", "is_low"])[col].transform(sum)

df["nilf_rate_cey"] = df["nilf_cey"] / (
    df["nilf_cey"] + df["unemp_cey"] + df["emp_cey"]
)
df["unemp_rate_cey"] = df["unemp_cey"] / (df["unemp_cey"] + df["emp_cey"])
df["ln_wage_cey"] = np.log(df["inc_cey"] / df["hours_cey"])
df.loc[df["ln_wage_cey"] == -np.inf, "ln_wage_cey"] = np.nan

cey_cols = ["pop_cey", "nilf_rate_cey", "unemp_rate_cey", "ln_wage_cey"]

# %%
# keep one observation per conspuma x education x year
to_keep = ["conspuma", "statefip", "is_low", "year", "mex_y", *cy_cols, *cey_cols]
df = df[to_keep].copy()
df.drop_duplicates(inplace=True)

# %%
# reshape to one obserevation per conspuma x education
cols = df.columns.to_list()  # save the column names
df = df.reset_index().pivot_table(
    index=["conspuma", "statefip", "is_low"], columns="year"
)

# %%
# get one observation per conspuma

df["dnilf_rate_ce"] = df["nilf_rate_cey", 2000] - df["nilf_rate_cey", 1990]
df["dunemp_rate_ce"] = df["unemp_rate_cey", 2000] - df["unemp_rate_cey", 1990]
df["dln_wage_ce"] = df["ln_wage_cey", 2000] - df["ln_wage_cey", 1990]

df.drop(["nilf_rate_cey", "unemp_rate_cey", "ln_wage_cey"], axis=1, inplace=True)

df = df.pivot_table(index=["conspuma", "statefip"], columns=["is_low"])

# %%
# create double differenced outcome variables (our Ys)

df["Dnilf_rate_c"] = df["dnilf_rate_ce", "", 0] - df["dnilf_rate_ce", "", 1]
df["Dunemp_rate_c"] = df["dunemp_rate_ce", "", 0] - df["dunemp_rate_ce", "", 1]
df["Dln_wage_c"] = df["dln_wage_ce", "", 0] - df["dln_wage_ce", "", 1]

df.drop(["dnilf_rate_ce", "dunemp_rate_ce", "dln_wage_ce"], axis=1, inplace=True)

# %%
# create our X and regression weight
df["weight"] = df["pop_cey", 1990, 0] + df["pop_cey", 1990, 1]
df["x"] = (
    df["pop_cey", 2000, 1] / (df["pop_cey", 2000, 0] + df["pop_cey", 2000, 1])
    - df["pop_cey", 1990, 1] / df["weight"]
)

df.drop("pop_cey", axis=1, inplace=True)

# %%
# create our instrument
df["z"] = (
    (1 / df["weight"])
    * (df["mex_cy", 1990, 0] / df["mex_y", 1990, 0])
    * (df["mex_y", 2000, 0] - df["mex_y", 1990, 0])
)

df.drop(["mex_cy", "mex_y"], axis=1, inplace=True)

# %%
df.head()

# %% [markdown]
# ### Now run regressions! Show the first stage for 'x', then second stage regressions for NILF, wage, and unemployment rates
