# Copyright (c) 2017-present, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


import pandas as pd
from forecaster import Structural

if __name__ == '__main__':
    df = pd.read_csv('./data/retail_sales.csv')
    model = Structural(monthly_changepoints=True,
                       yearly_seasonality=True,
                       weekly_seasonality=True).fit(df)
    fitted_y = model.predict(df)

    new_df = pd.DataFrame({'ds': model.make_forecast_dates(df, h=50)})
    forecast_y = model.predict(new_df)

    yhat = pd.concat([fitted_y, forecast_y]).reset_index(drop=True)
    yhat.plot()

