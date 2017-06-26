
data {
  int T;                                // sample size
  vector[T] t;                          // time indices
  vector[T] y;                          // time series values
}

parameters {
  real m;                               // slope
  real b;                               // intercept
  real<lower=0> sigma;                  // sd of observations
}

model {
  // likelihood
  y ~ normal(m * t + b, sigma);
}

