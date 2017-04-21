# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle

from pystan import StanModel

import numpy as np
import pandas as pd


class Structural(object):
    def __init__(self,
                 stan_model_filepath,
                 is_monthly_changepoints=True,
                 is_yearly_seasonality=True,
                 is_weekly_seasonality=True,
                 yearly_order=5,
                 weekly_order=3):

        # FILEPATH TO STAN MODEL (.PKL OR .STAN)
        self.model = self.import_stan_model(stan_model_filepath)

        # CHANGEPOINTS IN MEAN TREND
        self.is_monthly_changepoints = is_monthly_changepoints

        # SEASONALITY COMPONENT
        self.is_yearly_seasonality = is_yearly_seasonality
        self.is_weekly_seasonality = is_weekly_seasonality
        self.yearly_order = yearly_order
        self.weekly_order = weekly_order

        # DERIVED FROM `df` DURING PROCESSING PHASE OF fit()
        self.start_date = None
        self.end_date = None

        self.y_min = None
        self.y_max = None

        self.cpt_dates = None
        self.cpt_t = None

        self.last_observed_date = None

        # SET BY fit() IN STAN PHASE
        self.stan_fit_params = {}

    # --------------------------------------------
    #
    #                    API
    #
    # --------------------------------------------

    @staticmethod
    def create(name, **kwargs):
        possible_models = {
            'linear_trend': LinearTrend
        }

        if possible_models.get(name) is None:
            raise Exception('Model {} doesnt exist.'.format(name))
        else:
            model = possible_models[name](**kwargs)

        return model

    def fit(self, df):
        """Fits model parameters using Stan"""

        # PART 1: PROCESSING
        df = self.prepare_df(df)

        self.start_date = df['ds'].min()
        self.end_date = df['ds'].max()

        self.y_min = df['y'].min()
        self.y_max = df['y'].max()

        self.cpt_dates = self.generate_monthly_changepoints(dates=df['ds'])
        self.cpt_t = self.standardize_dates(dates=self.cpt_dates)

        self.last_observed_date = df['ds'].iloc[-1]

        t = self.standardize_dates(dates=df['ds'])
        y_scaled = self.standardize_y(y=df['y'])
        changepoint_df = self.make_changepoint_df(df)
        seasonality_df = self.make_seasonality_df(df)

        # PART 2: STAN
        self.stan_fit_params = self._fit_with_stan(t,
                                                   y_scaled,
                                                   changepoint_df,
                                                   seasonality_df)

        return self

    def _fit_with_stan(self, t, y_scaled, changepoint_df, seasonality_df):
        raise NotImplementedError

    def predict(self, new_df):
        """Predicts values at each `ds` date in `new_df`"""

        new_df = self.prepare_df(df=new_df)
        new_t = self.standardize_dates(dates=new_df['ds'])

        predicted_df = self._predict_standardized(new_df, new_t)
        predicted_df['yhat'] = self.unstandardize_y(y=predicted_df['yhat'])

        return predicted_df

    def _predict_standardized(self, new_df, new_t):
        raise NotImplementedError

    # --------------------------------------------
    #
    #                 IO METHODS
    #
    # --------------------------------------------

    @staticmethod
    def compile_stan_model(input_stan_filepath, output_model_filepath=None):
        with open(input_stan_filepath) as f:
            model_code = f.read()

        model = StanModel(model_code=model_code)

        if output_model_filepath is None:
            output_model_filepath = input_stan_filepath.replace('.stan',
                                                                '.pkl')
        with open(output_model_filepath, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def import_stan_model(stan_model_filepath):
        with open(stan_model_filepath, 'rb') as f:
            model = pickle.load(f)
        return model

    # --------------------------------------------
    #
    #      METHODS FOR DATA MANIPULATION
    #
    # --------------------------------------------

    @staticmethod
    def prepare_df(df):
        df = df.dropna()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        df = df.reset_index(drop=True)
        return df

    def standardize_dates(self, dates):
        return (dates - self.start_date) / (self.end_date - self.start_date)

    def standardize_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)

    def unstandardize_y(self, y):
        return y * (self.y_max - self.y_min) + self.y_min

    def make_forecast_dates(self, h):
        start_date = self.last_observed_date + pd.to_timedelta('1 day')
        return pd.date_range(start_date, periods=h)

    # --------------------------------------------
    #
    #      METHODS FOR SEASONALITY FEATURES
    #
    # --------------------------------------------

    @staticmethod
    def fourier_expansion(t, period, order):
        """Returns df of fourier terms evaluated at each `t`"""

        if order > period / 2:
            raise Exception('order should be <= period/2')

        fourier_term_matrix = [
            trig_fun((2.0 * np.pi * (k + 1) * t / period))
            for k in range(order)
            for trig_fun in (np.sin, np.cos)
            ]
        fourier_term_names = [
            '{}_{}'.format(fun_name, k + 1)
            for k in range(order)
            for fun_name in ('sin', 'cos')
            ]

        return pd.DataFrame(data=np.column_stack(fourier_term_matrix),
                            columns=fourier_term_names)

    def make_seasonality_df(self, df):
        """Returns df of fourier terms evaluated at each row of `df`"""

        # convert to days since epoch for consistent baseline
        t = np.array((df['ds'] - pd.Timestamp('1970-01-01')).dt.days)

        seasonality_features = []

        if self.is_yearly_seasonality:
            seasonality_features.append(
                self.fourier_expansion(t=t,
                                       period=365.25,
                                       order=self.yearly_order).rename(
                    columns=lambda name: 'yearly_{}'.format(name))
            )

        if self.is_weekly_seasonality:
            seasonality_features.append(
                self.fourier_expansion(t=t,
                                       period=7,
                                       order=self.weekly_order).rename(
                    columns=lambda name: 'weekly_{}'.format(name))
            )

        if len(seasonality_features) == 0:
            return pd.DataFrame(data={'seasonal_zeros': np.zeros(df.shape[0])})
        else:
            return pd.concat(seasonality_features, axis=1)

    # --------------------------------------------
    #
    #      METHODS FOR CHANGEPOINT FEATURES
    #
    # --------------------------------------------

    def make_changepoint_df(self, df):
        """Returns df of indicators evaluated at each row of `df`"""

        if self.is_monthly_changepoints:
            cpt_matrix = [df['ds'] >= d for d in self.cpt_dates]
            cpt_names = ['cpt_{}'.format(d.date()) for d in self.cpt_dates]
            return pd.DataFrame(data=np.column_stack(cpt_matrix).astype(float),
                                columns=cpt_names)
        else:
            return pd.DataFrame(data={'cpt_zeros': np.zeros(df.shape[0])})

    def generate_monthly_changepoints(self, dates):
        """Returns pd.Series of pd.Timestamps and floats"""

        if self.is_monthly_changepoints:
            return dates[dates.dt.is_month_start][1:-1]
        else:
            return pd.Series(dates[0])


class LinearTrend(Structural):
    """

    """

    def __init__(self,
                 slope_prior_sigma=5.0,
                 intercept_prior_sigma=5.0,
                 seasonality_prior_sigma=5.0,
                 changepoint_prior_sigma=0.5,
                 error_sd_prior_sigma=5.0,
                 **kwargs):

        self.name = 'linear_trend'

        self.slope_prior_sigma = slope_prior_sigma
        self.intercept_prior_sigma = intercept_prior_sigma
        self.changepoint_prior_sigma = float(changepoint_prior_sigma)
        self.seasonality_prior_sigma = float(seasonality_prior_sigma)
        self.error_sd_prior_sigma = error_sd_prior_sigma

        super(LinearTrend, self).__init__(**kwargs)

    def _fit_with_stan(self, t, y_scaled, changepoint_df, seasonality_df):
        """Returns dict of MAP estimates"""

        stan_data = {
            'T': y_scaled.size,
            't': t,
            'y': y_scaled,
            'tau': self.error_sd_prior_sigma,

            'sigma_m': self.slope_prior_sigma,
            'sigma_b': self.intercept_prior_sigma,

            'C': changepoint_df.shape[1],
            'cpt_t': self.cpt_t,
            'cpt_df': changepoint_df,
            'sigma_delta': self.changepoint_prior_sigma,

            'S': seasonality_df.shape[1],
            'X': seasonality_df,
            'sigma_beta': self.seasonality_prior_sigma
        }

        stan_init = lambda: {
            'm': y_scaled.iloc[-1] - y_scaled.iloc[0],
            'b': y_scaled.iloc[0],
            'delta': np.zeros(changepoint_df.shape[1]),
            'beta': np.zeros(seasonality_df.shape[1]),
            'sigma_y': 1.0,
        }

        stan_fit_params = self.model.optimizing(data=stan_data,
                                                init=stan_init,
                                                iter=1e4)
        for key, value in stan_fit_params.iteritems():
            stan_fit_params[key] = value.reshape(-1, )

        return stan_fit_params

    # TODO: CHANGE GAMMAS SO NOT SERIES. REQUIRES CHANGING CPT_T TO NUMPY
    def _predict_standardized(self, new_df, new_t):
        """Predicts values at each `ds` date in `new_df`"""

        # COMPUTE TREND OVER TIME
        gammas = -self.cpt_t * self.stan_fit_params['delta']
        m_t = self.stan_fit_params['m'].repeat(new_t.size)
        b_t = self.stan_fit_params['b'].repeat(new_t.size)
        for j, cpt_t in enumerate(self.cpt_t):
            index_t = new_t >= cpt_t
            m_t[index_t] += self.stan_fit_params['delta'][j]
            b_t[index_t] += gammas.iloc[j]
        mu_t = (m_t * new_t + b_t).rename('trend')

        # COMPUTE SEASONALITY COMPONENT OVER TIME
        s_t = (self.make_seasonality_df(new_df)
               .dot(self.stan_fit_params['beta'])
               .rename('seasonality'))

        # ELEMENT-WISE SUM OF SERIES
        yhat_t = (mu_t + s_t).rename('yhat')

        return pd.concat([new_df, yhat_t], axis=1)



