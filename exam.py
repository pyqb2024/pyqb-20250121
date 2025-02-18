# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: January 21, 2025
#
# You can solve the exercises below by using standard Python 3.12 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.12/), [NumPy](https://numpy.org/doc/1.26/index.html), [Matplotlib](https://matplotlib.org/3.10.0/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/version/2.2/index.html), [PyMC](https://www.pymc.io/projects/docs/en/stable/api.html).
# You can also look at the [slides](https://homes.di.unimi.it/monga/lucidi2425/pyqb00.pdf) or your code on [GitHub](https://github.com).
#
# **It is forbidden to communicate with others or "ask questions" online (i.e., stackoverflow is ok if the answer is already there, but you cannot ask a new question or use ChatGPT and similar products)**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
#

import numpy as np
import pandas as pd             # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm               # type: ignore
import arviz as az              # type: ignore

# ### Exercise 1 (max 2 points)
#
# The file [bird_foo_data.csv](./bird_foo_data.csv) dataset contains the data from all fox diet studies that were analysed to generate the frequency of occurrence of birds in fox diets across Australia. The data come from: 
# Stobo-Wilson, A., Murphy, B., Legge, S., Caceres-Escobar, H., Chapple, D., Crawford, H., Dawson, S., Dickman, C., Doherty, T., Fleming, P., Garnett, S., Gentle, M., Newsome, T., Palmer, R., Rees, M., Ritchie, E., Speed, J., Stuart, J.-M., Suarez-Castro, A., … Woinarski, J. (2022). Counting the bodies: estimating the numbers and spatial variation of Australian reptiles, birds and mammals killed by two invasive mesopredators [Zenodo](https://doi.org/10.5061/dryad.bk3j9kdcz)
#
# Read the dataset in a DataFrame `birds`, use the ID column as the index.
#
#
# Columns: subheadings:
# - taxon: the taxon group that the dataset relates to
# - type: specifies whether the diet data was collected from fox scat or fox stomach samples or both
# - p_scat: the proportion of fox diet data that was collected from fox scats relative to fox stomachs
# - samples: the number of individual fox diet samples that were collected
# - total_in: the total number of individuals identified from fox stomach diet samples 
# - mean_in: the mean number of individuals identified per fox stomach from all fox diet stomach 	samples
# - map: mean annual precipitation at the study location where fox diet samples were collected 
# - mat: mean annual temperature at the study location where fox diet samples were collected
# - cover: mean tree cover at the study location where fox diet samples were collected
# - rugged: topographic ruggedness at the study location where fox diet samples were collected
# - popden: human population density at the study location where fox diet samples were collected 
# - vast: vegetation disturbance at the study location where fox diet samples were collected
# - distcoast: distance to coast at the study location where fox diet samples were collected 
#

pass

# ### Exercise 2 (max 4 points)
#
# Compute the number of samples collected in the scat and in the stomach. When the `type` column is "both (not stated)" or "scat & stomach" use the value of `p_scat` to distribuite the samples between the two types (float values are ok). To get the full marks do not use explicit loops.

pass

# ### Exercise 3 (max 7 points)
#
# Define a function which takes a takes a `pd.Series` (of integers) and it compute a (shorter) one with the sums of subsequent triplets of values.
# For example if the series contains the values 1,2,3,4,5 the result should be a series with 6,9,12.
#
# To get the full marks, you should declare correctly the type hints (the signature of the function) and add a doctest string.

pass

# ### Exercise 4 (max 2 points)
#
# Apply the function define in Exercise 3 to column `samples`. To get the full marks do not use explicit loops.

pass

# ### Exercise 5 (max 5 points)
#
# Make a copy of the DataFrame `birds` in which every row with `type` "both (not stated)" or "scat & stomach" is duplicated: one row should have
# type "scat" and the other "stomach", the `samples` and `total_in` (if is not n/a) are distributed according to `p_scat`. Use only integer numbers for the samples and be sure the sum of samples is preserved for each pair of duplicated rows.

pass

# ### Exercise 6 (max 4 points)
#
# Add to the DataFrame birds a column with the standardized value of `total_in`. Remember that the standardized value measures how many standard deviations a specific value is far from the mean. If you have a ndarray of values `xx`: `(xx - xx.mean())/xx.std()`. Then plot a density histogram of this new column. 

pass


# ### Exercise 7 (max 4 points)
#
#
# Plot a matrix of scatter plots (for each pair a,b you can plot just a,b and leave b,a empty) of all the combinations of `map`, `mat`, `cover`, `ragged`. They should appear all in the same figure. Put also a proper title to each plot.

pass

# ### Exercise 8 (max 5 points)
#
# Consider this statistical model: 
#
# - the parameter $\alpha$ is normally distributed with mean 0, and stdev 3
# - the parameter $\beta$ is normally distributed with mean 0, and stdev 5
# - the parameter $\sigma$ is exponentially distributed with $\lambda = 1$
# - the the mean of the observed value of `map` is given by $\alpha + \beta\cdot C$ where C is the observed value of `cover`, its std deviation is $\sigma$
#
# Use PyMC to sample the posterior distributions after having seen the actual values for `map`.  Plot the posterior.

pass


