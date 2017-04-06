/*
*  Copyright (c) 2017, Kyle Lo
*  All rights reserved.
*
*  This source code is licensed under the BSD-style license found in the
*  LICENSE file in the root directory of this source tree.
*/


data {
  int T;                                // sample size
  vector[T] t;                          // time indices
  vector[T] y;                          // time series values

  real<lower=0> sigma_m;                // known sd on `m` prior
  real<lower=0> sigma_b;                // known sd on `b` prior

  int<lower=1> C;                       // number of changepoints
  vector[C] cpt_t;                      // changepoint time indices
  matrix[T, C] cpt_df;                  // changepoint indicator features
  real<lower=0> sigma_delta;            // known scale param on `delta` prior

  int<lower=1> S;                       // number of seasonality features
  matrix[T, S] X;                       // seasonality features
  real<lower=0> sigma_beta;             // known sd on `beta` prior
}

parameters {
  real<lower=0, upper=1> p;             // prob of one class vs other
  real m;                               // base slope
  real b;                               // base intercept
  vector[C] delta;                      // changepoint effects (slope changes)
  vector[S] beta;                       // seasonality effects
}

transformed parameters {
  vector[C] gamma;                      // changes to intercept at changepoints

  for (i in 1:C) {
    gamma[i] = -cpt_t[i] * delta[i];
  }
}

model {
  //priors
  m ~ normal(0, sigma_m);
  b ~ normal(0, sigma_b);
  delta ~ double_exponential(0, sigma_delta);
  beta ~ normal(0, sigma_beta);

  // Likelihood
  for (i in 1:T) {
    (y[i] == 0) ~ bernoulli(p);
    if (y[i] > 0)
      y[i] ~ poisson((m + cpt_df[i] * delta) * t[i] + (b + cpt_df[i] * gamma) + X * beta) T[1, ];
  }
}

