# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from forecaster import LinearTrend
import pkg_resources

if __name__ == '__main__':
    # --------------------------------------------
    #
    #                  EXAMPLE 1
    #
    # --------------------------------------------

    # load data
    df = pd.read_csv('retail_sales.csv')

    # first time instantiating Structural compiles Stan model
    stan_model_filepath = pkg_resources.resource_filename('forecaster',
                                                          'stan_models/linear.stan')
    model = LinearTrend(stan_model_filepath,
                        monthly_changepoints=True,
                        yearly_seasonality=True,
                        weekly_seasonality=True).fit(df)

    # get fitted values
    fitted_yhat = model.predict(df)
    print(fitted_yhat.head())

    # forecast next 50 days
    new_df = pd.DataFrame({'ds': model.make_forecast_dates(df, h=50)})
    forecasted_yhat = model.predict(new_df)
    print(forecasted_yhat.head())

    # --------------------------------------------
    #
    #                  EXAMPLE 2
    #
    # --------------------------------------------

    # for subsequent runs, use pickled stan model
    stan_model_filepath = stan_model_filepath.replace('.stan', '.pkl')

    model2 = LinearTrend(stan_model_filepath,
                         monthly_changepoints=False,
                         yearly_seasonality=False,
                         weekly_seasonality=False).fit(df)

    fitted_yhat2 = model2.predict(df)
    print(fitted_yhat2.head())
