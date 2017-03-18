## Forecaster for Structural Time Series model forecasting

This repo contains a "lite" version of a structural time series model provided in Facebook's Prophet library for Python.  I'm planning on extending this module in a different direction, but `example.py` shows that the base functionality (univariate time-series forecasting of daily data) hasn't changed.  See the original repo here: https://github.com/facebookincubator/prophet.  This code is released under the BSD 3-Clause license, which I've included in this repo under `LICENSE`.

#### Compared to Prophet

This library provides a `Structural` class that implements Prophet's structural time series model with "linear growth" trend.  The original Stan model definition has not changed (see `linear.stan`).

Major differences between the `Structural` (here) and `Prophet` (original) classes include:
  
  - Prophet compiles its Stan models upon installation and loads them in `Prophet.fit()`. This library compiles and saves its Stan models when `Structural.fit()` is called for the first time, and future calls will use the saved copy.  This makes model development easier: For example, adding an option to force the Stan model to recompile at run-time is more convenient than reinstalling the library each time I modify the `.stan` file.
  - `Prophet.set_changepoints()` generated changepoints using `np.linspace` over the first 80% of training set dates.  `Structural.generate_changepoints()` changepoints are placed at the beginning of each month (excluding the first and last) in the training data.
  - Prophet's model includes indicator features for holidays while the model in `Structural` doesn't have these.
  - `Prophet` allows the user to provide known changepoints while `Structural` doesn't.

Minor differences include:

  - `Prophet.setup_dataframe()` would return a cleaned version of the input dataframe, but it would also set the values of `self.y_scale`, `self.start`, etc..  In `Structural`, the only method that can set members is `fit()` and all other methods will return a result, so there's less worry about side effects.
  - Missing data removal moved from `Prophet.fit()` to `Structural.clean_df()` along with other dataframe-processing operations from `Prophet.setup_dataframe()`.
  - The user can now specify `yearly_order` and `monthly_order` for the fourier expansion when instantiating a `Structural` object.  In Prophet, these orders are fixed to `10` and `3` within `Prophet.make_all_seasonality_features()` so can't be tuned.
  - `Prophet.make_all_seasonality_features()` always creates a `zeros` feature vector, even when seasonality is specified.  This is stylistically inconsistent with `Prophet.set_changepoints()`, which only creates a dummy changepoint when no changepoints are specified.  `Structural.make_seasonality_df()` only creates the `zeros` feature vector when no seasonality is specified.
  - When specifying no changepoints, `Prophet.get_changepoint_matrix()` will still return a feature vector of ones.  This results in fitting a non-zero `delta` term that's fixed in `Prophet.fit()` by setting `k = k + delta` and `delta = 0`.  This situation is dealt with cleanly in `Structural.make_changepoint_df()` by returning a feature vector of zeros for the no-changepoint case.  Then there need not be any post-fitting correction of `k` and `delta`.
  
I've currently removed support for these functions:

  1. MCMC sampling
  2. Variability estimation (based on (1))
  3. Plotting

#### Installation

```
git clone https://github.com/kyleclo/forecaster.git
pip install -r requirements.txt
```

####

`example.py` contains an example of using this library on the retail sales example dataset provided in the Prophet repo.  The defaults of `Structural` should look very similar to the defaults of `Prophet` other than the generated changepoints.


#### Work in Progress

I'm currently working on (in no particular order):

  1. Adding in MCMC sampling + variability estimation
  2. Adding in manual changepoint selection
  3. Adding in holiday indicators
  4. Adding a prior over changepoint locations
  5. Adding in ARMA errors
  6. Adding a testing suite
  7. Creating a pypi distribution
  


