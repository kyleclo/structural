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

        # PARAMETERS THAT DEPEND ON `df` SET BY fit() IN PROCESSING PHASE
        self.start_date = None
        self.end_date = None
        self.y_max = None
        self.cpt_dates = None
        self.cpt_t = None

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

        raise NotImplementedError

    def predict(self, new_df):
        """Predicts values at each `ds` date in `new_df`"""

        raise NotImplementedError

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
        return y / self.y_max

    # TODO: change to take last_observed_date
    @staticmethod
    def make_forecast_dates(df, h):
        df = Structural.prepare_df(df)
        last_observed_date = df.iloc[-1]['ds']
        start_date = last_observed_date + pd.to_timedelta('1 day')
        return pd.date_range(start_date, periods=h)

    # --------------------------------------------
    #
    #      METHODS FOR SEASONALITY FEATURES
    #
    # --------------------------------------------

    @staticmethod
    def fourier_expansion(t, period, order):
        """Returns df of fourier terms evaluated at each `t`"""

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

    def make_seasonality_df(self, dates):
        """Returns df of fourier terms """

        # convert to days since epoch for consistent baseline
        t = np.array((dates - pd.Timestamp('1970-01-01')).dt.days)

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
            return pd.DataFrame(data={'seasonal_zeros': np.zeros(dates.size)})
        else:
            return pd.concat(seasonality_features, axis=1)

    # --------------------------------------------
    #
    #      METHODS FOR CHANGEPOINT FEATURES
    #
    # --------------------------------------------

    def make_changepoint_df(self, dates):
        """Returns df of indicators for each changepoint (column) at `dates`"""

        if self.is_monthly_changepoints:
            cpt_matrix = [dates >= d for d in self.cpt_dates]
            cpt_names = ['cpt_{}'.format(d.date()) for d in self.cpt_dates]
            return pd.DataFrame(data=np.column_stack(cpt_matrix).astype(float),
                                columns=cpt_names)
        else:
            return pd.DataFrame(data={'cpt_zeros': np.zeros(dates.size)})

    def generate_monthly_changepoints(self, dates):
        """Returns pd.Series of pd.Timestamps and floats"""

        if self.is_monthly_changepoints:
            return dates[dates.dt.is_month_start][1:-1]
        else:
            return pd.Series(dates[0])

    # --------------------------------------------
    #
    #                 IO METHODS
    #
    # --------------------------------------------

    @staticmethod
    def compile_stan_model(stan_model_filepath):
        with open(stan_model_filepath) as f:
            model_code = f.read()

        model = StanModel(model_code=model_code)

        with open(stan_model_filepath.replace('.stan', '.pkl'), 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def import_stan_model(stan_model_filepath):
        with open(stan_model_filepath, 'rb') as f:
            model = pickle.load(f)
        return model


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

    def fit(self, df):
        # PART 1: PROCESSING
        df = self.prepare_df(df)

        self.start_date = df['ds'].min()
        self.end_date = df['ds'].max()
        self.y_max = df['y'].max()

        t = self.standardize_dates(dates=df['ds'])
        y_scaled = self.standardize_y(y=df['y'])

        self.cpt_dates = self.generate_monthly_changepoints(dates=df['ds'])
        self.cpt_t = self.standardize_dates(dates=self.cpt_dates)

        changepoint_df = self.make_changepoint_df(dates=df['ds'])
        seasonality_df = self.make_seasonality_df(dates=df['ds'])

        # PART 2: STAN
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

        self.stan_fit_params = self.model.optimizing(data=stan_data,
                                                     init=stan_init,
                                                     iter=1e4)
        for key, value in self.stan_fit_params.iteritems():
            self.stan_fit_params[key] = value.reshape(-1, )

        return self

    def predict(self, new_df):
        """Predicts values at each `ds` date in `new_df`"""

        new_df = self.prepare_df(new_df)
        new_t = self.standardize_dates(dates=new_df['ds'])

        # COMPUTE TREND OVER TIME
        gammas = -self.cpt_t * self.stan_fit_params['delta']
        m_t = self.stan_fit_params['m'].repeat(new_t.size)
        b_t = self.stan_fit_params['b'].repeat(new_t.size)
        for j, cpt_t in enumerate(self.cpt_t):
            index_t = new_t >= cpt_t
            m_t[index_t] += self.stan_fit_params['delta'][j]
            b_t[index_t] += gammas.iloc[j]
        trend = ((m_t * new_t + b_t) * self.y_max).rename('trend')

        # COMPUTE SEASONALITY COMPONENT OVER TIME
        seasonality = (self.make_seasonality_df(new_df['ds']).dot(
            self.stan_fit_params['beta']) * self.y_max).rename('seasonality')

        yhat = (trend + seasonality).rename('yhat')

        # return pd.concat([new_df, yhat, trend, seasonality], axis=1)
        return pd.concat([new_df, yhat], axis=1)
