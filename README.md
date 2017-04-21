# Structural time series modeling and forecasting

**Structural** is a Python library for structural time series modeling and forecasting of daily univariate sequential data.

This code is released under the BSD 3-Clause license, which I've included in this repo under `/LICENSE`.


## Installation

Clone the repo if you want the source code
```
git clone https://github.com/kyleclo/structural.git
pip install -r requirements.txt
```

Or install the module using
```
pip install git+git://github.com/kyleclo/structural.git#egg=structural
```


## Usage

Typical usage looks like this:

```python
import pandas as pd
from structural import LinearTrend

df = pd.read_csv(DATA_FILEPATH)
model = LinearTrend(STAN_MODEL_FILEPATH)
model.fit(df)

yhat_fitted = model.predict(df)

new_df = pd.DataFrame({'ds': model.make_forecast_dates(h=100)})
yhat_forecasted = model.predict(new_df)
```

See `/example.py` for a sample script.  I recommend using it as starter code for those looking to deploy an automated forecasting service.


## Backlog

I'm currently working on (in no particular order):

  1. Adding in MCMC sampling + variability estimation
  2. Adding in manual changepoint selection
  3. Adding in manual holiday indicators
  4. Adding a prior over changepoint locations
  5. Adding in ARMA errors
  6. Adding a testing suite
  7. Creating a pypi distribution
  8. Generalizing LinearTrend to allow for link functions
  9. Refactor code structure so we're "building" each structural component into the model.  (This probably requires piecing together Stan code snippets at runtime)
  

## About

This project was inspired by **Prophet**, an open-source library released by the good folks over at Facebook.  Check out their repo here: https://github.com/facebookincubator/prophet.

I developed this library while working at CDK Global on some projects involving large-scale time series forecasting.  I tried using Prophet at first, but their library seems more geared toward use by analysts who will be tuning models manually.  I needed a more general-purpose, automated procedure for my project, so I ended up writing this library instead.  I've documented the differences below, but overall I've maintained a similar API.

## Compared to Prophet

Major differences include:
  
  - In Structural, `Structural` is actually an abstract base class.  Users implement subclasses of `Structural` and instantiation is handled by `Structural.create()` at runtime.  I found this more extensible than Prophet's `Prophet` class, which uses if-statements in its methods to switch between "linear" and "logistic" growth models.    
  
  - In Structural, Stan models are compiled and imported using `Structural.compile_stan_model()` and `Structural.import_stan_model()`, as opposed to Prophet, which compiles Stan models during package installation.  I made this change because I wanted the flexibility to add / modify / select different Stan models at runtime, without having to re-run the package installation.
  
  - In Structural, automated changepoint generation places changepoints at the first of each month (excluding the first and last) in the training data, as opposed to Prophet, which generates changepoints using `np.linspace` over the first 80% of training set dates.  I simply felt this was a more intuitive default choice when no changepoints are specified.

Other differences are more for style / personal preference:

  - Stylistically, I've written Structural such that the only methods that can set members are the constructor and `Structural.fit()`, while all other methods will return a result.  Hence, I've removed/rewritten methods with side-effects like `Prophet.setup_dataframe()` and `Prophet.set_changepoints()`.
  
  - The user can now specify `yearly_order` and `monthly_order` for the fourier expansion when instantiating a `Structural` object.  In Prophet, these values are hard-coded to `10` and `3` within the `Prophet.make_all_seasonality_features()` method.
  
  - I've rewritten `Structural.make_seasonality_df()` and `Structural.make_changepoint_df()` to have consistent style: They both only create the `zeros` feature vector when specifying no seasonality or no changepoints, respectively.  
    
  - In Structural, `Structural.make_changepoint_df()` returns a feature vector of zeros (instead of ones) in the no-changepoint case, so the fitted `delta` will be zero.  In Prophet, when specifying no changepoints, `Prophet.get_changepoint_matrix()` will still return a feature vector of ones.  This results in fitting a non-zero `delta` term, which requires an additional correction step in `Prophet.fit()` by setting `k = k + delta` and `delta = 0`.  
  
  
Structural currently doesn't have support for:

  1. MCMC sampling
  2. Variability estimation (based on (1))
  3. Plotting   

which Prophet provides.


*Note: I'm making these comparisons based on what I saw in Prophet v0.0.post1 which have changed by now*