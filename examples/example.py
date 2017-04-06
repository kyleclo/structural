# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from structural import Structural, LinearTrend

import matplotlib.pyplot as plt

import os
import pkg_resources

STAN_DIRPATH = pkg_resources.resource_filename('structural', 'stan_models')
STAN_MODEL_NAME = 'linear_trend'
DATA_FILEPATH = os.path.join(os.path.dirname(__file__), 'retail_sales.csv')

if __name__ == '__main__':
    # --------------------------------------------
    #
    #                  EXAMPLE 1
    #
    # --------------------------------------------

    # load data
    df = pd.read_csv(DATA_FILEPATH)

    # If pickled Stan model not exist, then compile one.  Use if exists.
    stan_model_filepath = os.path.join(STAN_DIRPATH,
                                       '{}.pkl'.format(STAN_MODEL_NAME))
    if not os.path.exists(stan_model_filepath):
        Structural.compile_stan_model(stan_model_filepath.replace('.pkl',
                                                                  '.stan'))

    # fit model
    model = LinearTrend(stan_model_filepath=stan_model_filepath).fit(df)

    # get fitted values
    yhat_fitted = model.predict(df)

    # forecast next 1000 days
    new_df = pd.DataFrame({'ds': model.make_forecast_dates(h=1000)})
    yhat_forecasted = model.predict(new_df)

    # plot
    plt.close()
    pd.concat([yhat_fitted.set_index('ds'),
               yhat_forecasted.set_index('ds')], axis=0).plot()
    plt.show()

    # # --------------------------------------------
    # #
    # #                  EXAMPLE 2
    # #
    # # --------------------------------------------
    #
    # # it's possible to change arguments while using same pickled Stan model
    # # for example, let's use Prophet's defaults for `*_order` and `*_sigma`
    # model2 = LinearTrend(stan_model_filepath=stan_model_filepath,
    #                      is_monthly_changepoints=True,
    #                      is_yearly_seasonality=True,
    #                      is_weekly_seasonality=True,
    #                      yearly_order=10,
    #                      weekly_order=3,
    #                      slope_prior_sigma=5.0,
    #                      intercept_prior_sigma=5.0,
    #                      seasonality_prior_sigma=10.0,
    #                      changepoint_prior_sigma=0.05,
    #                      error_sd_prior_sigma=0.5).fit(df)
    #
    # # --------------------------------------------
    # #
    # #                  EXAMPLE 3
    # #
    # # --------------------------------------------
    #
    # # `Structural` also provides a method for choosing subclass at runtime
    # params = {
    #     'stan_model_filepath': stan_model_filepath,
    #     'is_monthly_changepoints': False,
    #     'is_yearly_seasonality': False,
    #     'is_weekly_seasonality': False
    # }
    # model3 = Structural.create(name=STAN_MODEL_NAME, **params).fit(df)
