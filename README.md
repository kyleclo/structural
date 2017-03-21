# Forecaster for Structural Time Series model forecasting

This Python module contains a "lite" version of the structural time series model implemented in Prophet, an open-source library released by Facebook.  See the original repo here: https://github.com/facebookincubator/prophet.

Like Prophet, this module is for performing univariate time-series forecasting of daily data.

This code is released under the BSD 3-Clause license, which I've included in this repo under `/LICENSE`.  

## Compared to Prophet

This module provides a `Structural` class that implements Prophet's structural time series model with "linear growth" trend, aka `Prophet(growth='linear')`.  The original Stan model definition has not changed (see `/stan_models/linear.stan`).

Major differences between the `Structural` (this) and `Prophet` (original) classes include:
  
  - Now, Stan models are compiled and saved when the `Structural` constructor is called for the first time, and future instantiations of the same model will load the pickled Stan model.  This is as opposed to Prophet, which compiles its Stan models during package installation.  I made this change so I could experiment with modifications to the Stan file without having to rerun the entire installation.
  
  - In this module, automated changepoint generation places changepoints at the first of each month (excluding the first and last) in the training data.  This is in contrast to Prophet, which generates changepoints using `np.linspace` over the first 80% of training set dates.
  
  - `Structural` (currently) doesn't support user specification of known changepoints or holidays.

Minor differences include:

  - Stylistically, I wrote `Structural` such that the only methods that can set members are the constructor and `Structural.fit()`, while all other methods will return a result.  Hence, I've removed/rewritten methods with side-effects like `Prophet.setup_dataframe()` and `Prophet.set_changepoints()`.
  
  - The user can now specify `yearly_order` and `monthly_order` for the fourier expansion when instantiating a `Structural` object.  In `Prophet`, these values are hard-coded to `10` and `3` within the `Prophet.make_all_seasonality_features()` method.
  
  - I've rewritten `Structural.make_seasonality_df()` and `Structural.make_changepoint_df()` to have consistent style: They both only create the `zeros` feature vector when no seasonality or no changepoints is specified, respectively.  
    
  - In `Prophet`, when specifying no changepoints, `Prophet.get_changepoint_matrix()` will still return a feature vector of ones.  This results in fitting a non-zero `delta` term, which requires fixing in `Prophet.fit()` by setting `k = k + delta` and `delta = 0`.  I've rewritten this so that `Structural.make_changepoint_df()` returns a feature vector of zeros (instead of ones) in the no-changepoint case so the fitted `delta` will be zero.
  
I've currently removed support for these functions:

  1. MCMC sampling
  2. Variability estimation (based on (1))
  3. Plotting

See the [backlog](#backlog) since I'm planning on adding back support for some of these.

## Installation

Clone the repo if you want the source code
```
git clone https://github.com/kyleclo/forecaster.git
pip install -r requirements.txt
```

Or install the module using
```
pip install git+git://github.com/kyleclo/forecaster.git#egg=forecaster
```

## Usage

`/example.py` contains an example of using this library on the retail sales example dataset provided in the Prophet repo.  The defaults of `Structural` should look very similar to the defaults of `Prophet` other than the generated changepoints.


## Backlog

I'm currently working on (in no particular order):

  1. Adding in MCMC sampling + variability estimation
  2. Adding in manual changepoint selection
  3. Adding in holiday indicators
  4. Adding a prior over changepoint locations
  5. Adding in ARMA errors
  6. Adding a testing suite
  7. Creating a pypi distribution
  


