# Copyright (c) 2017-present, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
from pystan import StanModel

import numpy as np
import pandas as pd


class Structural(object):
    def __init__(
            self,
            stan_model_filepath,
            monthly_changepoints=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            yearly_order=10,
            weekly_order=3,
            seasonality_prior_sigma=10.0,
            changepoint_prior_sigma=0.05
    ):
        # FILEPATH TO STAN MODEL (.PKL OR .STAN)
        self.model = self.import_stan_model(stan_model_filepath)

        # CHANGEPOINTS IN MEAN TREND
        self.is_monthly_changepoints = monthly_changepoints
        self.changepoint_prior_sigma = float(changepoint_prior_sigma)

        # SEASONALITY COMPONENT
        self.is_yearly_seasonality = yearly_seasonality
        self.is_weekly_seasonality = weekly_seasonality
        self.yearly_order = yearly_order
        self.weekly_order = weekly_order
        self.seasonality_prior_sigma = float(seasonality_prior_sigma)

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

    def fit(self, df):
        """Fits model parameters using Stan"""

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
            'S': changepoint_df.shape[1],
            'K': seasonality_df.shape[1],

            'y': y_scaled,
            't': t,

            'X': seasonality_df,
            'A': changepoint_df,
            't_change': self.cpt_t,

            'tau': self.changepoint_prior_sigma,
            'sigma': self.seasonality_prior_sigma
        }

        stan_init = lambda: {
            'k': y_scaled.iloc[-1] - y_scaled.iloc[0],
            'm': y_scaled.iloc[0],
            'delta': np.zeros(changepoint_df.shape[1]),
            'beta': np.zeros(seasonality_df.shape[1]),
            'sigma_obs': 1.0,
        }

        self.stan_fit_params = dict(self.model.optimizing(data=stan_data,
                                                          init=stan_init,
                                                          iter=1e4))
        return self

    def predict(self, new_df):
        """Predicts values at each `ds` date in `new_df`"""

        new_df = self.prepare_df(new_df)
        new_t = self.standardize_dates(dates=new_df['ds'])

        # k_t = self.stan_fit_params['k'] + np.matmul(
        #     self.make_changepoint_df(new_df['ds']),
        #     self.stan_fit_params['delta'])
        # m_t = self.stan_fit_params['m'] + np.matmul(
        #     self.make_changepoint_df(new_df['ds']),
        #     - self.cpt_t * self.stan_fit_params['delta'])

        # COMPUTE TREND OVER TIME
        gammas = -self.cpt_t * self.stan_fit_params['delta']
        k_t = self.stan_fit_params['k'].repeat(new_t.size)
        m_t = self.stan_fit_params['m'].repeat(new_t.size)
        for j, cpt_t in enumerate(self.cpt_t):
            index_t = new_t >= cpt_t
            k_t[index_t] += self.stan_fit_params['delta'][j]
            m_t[index_t] += gammas.iloc[j]
        trend = ((k_t * new_t + m_t) * self.y_max).rename('trend')

        # COMPUTE SEASONALITY COMPONENT OVER TIME
        seasonality = (self.make_seasonality_df(new_df['ds']).dot(
            self.stan_fit_params['beta']) * self.y_max).rename('seasonality')

        yhat = (trend + seasonality).rename('yhat')

        # return pd.concat([new_df, yhat, trend, seasonality], axis=1)
        return pd.concat([new_df, yhat], axis=1)

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
            return pd.Series()

    # --------------------------------------------
    #
    #                 IO METHODS
    #
    # --------------------------------------------

    @staticmethod
    def import_stan_model(stan_model_filepath):
        if not os.path.exists(stan_model_filepath):
            raise IOError('Invalid path to stan model.')

        file_extension = stan_model_filepath.split('.')[-1]
        if file_extension == 'pkl':
            with open(stan_model_filepath, 'rb') as f:
                model = pickle.load(f)
        elif file_extension == 'stan':
            with open(stan_model_filepath) as f:
                model_code = f.read()
            model = StanModel(model_code=model_code)
            with open(stan_model_filepath.replace('.stan', '.pkl'), 'wb') as f:
                pickle.dump(model, f)
        else:
            raise IOError('Input should be a .stan or .pkl file.')

        return model
