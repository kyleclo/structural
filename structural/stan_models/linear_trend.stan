/*
*  This is a refactored version of 'linear growth' model in Prophet by Facebook.
*  See their repo here: https://github.com/facebookincubator/prophet
*
*  This source code is licensed under the BSD-style license found in the
*  LICENSE file in the root directory of this source tree.
*/


data {
  int N;                                // sample size
  vector[N] t;                          // time indices
  vector[N] y;                          // time series values

  real<lower=0> sigma_m;                // known sd on `m` prior
  real<lower=0> sigma_b;                // known sd on `b` prior

  int<lower=1> C;                       // number of changepoints
  vector[C] cpt_t;                      // changepoint time indices
  matrix[N, C] cpt_df;                  // changepoint indicator features
  real<lower=0> sigma_delta;            // known scale param on `delta` prior

  int<lower=1> S;                       // number of seasonality features
  matrix[N, S] X;                       // seasonality features
  real<lower=0> sigma_beta;             // known sd on `beta` prior

  real<lower=0> tau;                    // known scale param on `sigma_y` prior
}

parameters {
  real m;                               // base slope
  real b;                               // base intercept
  vector[C] delta;                      // changepoint effects (slope changes)
  vector[S] beta;                       // seasonality effects
  real<lower=0> sigma_y;                // sd of observations
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
  sigma_y ~ cauchy(0, tau);

  // Likelihood
  y ~ normal((m + cpt_df * delta) .* t + (b + cpt_df * gamma) + X * beta, sigma_y);
}

