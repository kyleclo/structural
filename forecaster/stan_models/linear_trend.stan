data {
  int T;                                // sample size
  vector[T] t;                          // time indices
  vector[T] y;                          // time series values

  // Stan manual Sec 5.2 recommend default=5.0 if data standardized
  real<lower=0> sigma_m;                // known sd on `m` prior
  real<lower=0> sigma_b;                // known sd on `b` prior

  int C;                                // number of changepoints
  real cpt_t[C];                        // changepoint time indices
  matrix[T, C] cpt_df;                  // changepoint indicator features
  real<lower=0> sigma_delta;            // known scale param on `delta` prior (Prophet default=0.05)

  int<lower=1> S;                       // number of seasonality features
  matrix[T, S] X;                       // seasonality features
  real<lower=0> sigma_beta;             // known sd on `beta` prior (Prophet default=10.0)

  // Prophet provided default=0.5
  real<lower=0> tau;                    // known sd on `sigma_y` prior
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
  sigma_y ~ normal(0, tau);

  // Likelihood
  y ~ normal((m + cpt_df * delta) .* t + (b + cpt_df * gamma) + X * beta, sigma_y);
}
