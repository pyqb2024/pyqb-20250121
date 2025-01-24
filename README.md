# Fox diet

The file [bird_foo_data.csv](./bird_foo_data.csv) dataset contains the data from all fox diet studies that were analysed to generate the frequency of occurrence of birds in fox diets across Australia. The data come from:
Stobo-Wilson, A., Murphy, B., Legge, S., Caceres-Escobar, H., Chapple, D., Crawford, H., Dawson, S., Dickman, C., Doherty, T., Fleming, P., Garnett, S., Gentle, M., Newsome, T., Palmer, R., Rees, M., Ritchie, E., Speed, J., Stuart, J.-M., Suarez-Castro, A., … Woinarski, J. (2022). Counting the bodies: estimating the numbers and spatial variation of Australian reptiles, birds and mammals killed by two invasive mesopredators [Zenodo](https://doi.org/10.5061/dryad.bk3j9kdcz)



## Setup

The `exam.py` is intended to be modified as a
[jupyter](https://jupyter.org/) notebook. The notebook format `.ipynb`, however,
is not very convenient for configuration management, testing, and giving
feedback via the usual "pull-request" mechanism provided by Github. Thus, this
repository uses
[jupytext](https://jupytext.readthedocs.io/en/latest/install.html) to **pair** a
pure Python file with a notebook with the same name. The notebook is
automatically created when you open the Python file with jupyter, and the two
files are kept in sync. Do not add `exercise.ipynb` to the files managed by git.

To start, you need the following actions:

```sh
python3.12 -m venv VIRTUAL_ENVIRONMENT
# remember to activate the virtual environment according to your operating system rules:
# source VIRTUAL_ENVIRONMENT/bin/activate # adapt to your system
pip install -r requirements.txt
jupyter notebook
```

Then you can open the `exam.py` as a notebook in the browser.


## Test

To test examples in docstrings use:

```python
import doctest
doctest.testmod()
```


You can execute tests locally on the python file:


```sh
mypy exam.py
python -m doctest exam.py
```
