import numpy as np
from scipy.stats import norm

# The Geometric Brownian Motion (GBM) class simulates stock prices
class GBM:
    def __init__(self):
        # TODO: Replace the following with your code
        # set parameters to NaN
        self.mu = np.nan
        self.sigma = np.nan
        self.rng = np.random.default_rng()
        
    def simulate(self, N, K, Dt, S0):
        # N : time steps
        # K : number of trajectories
        traj = np.full((N+1, K), np.nan)
        # TODO: Your code goes here
        if not isinstance(S0, (int, float)):
            S0 = float(S0)
        drift = (self.mu - 0.5 * self.sigma**2) * np.linspace(Dt, N*Dt, N)
        for i in range(K):
            traj[0,i] = S0
            W = np.cumsum(norm.rvs(scale=np.sqrt(Dt), size=N))
            traj[1:,i] = S0 * np.exp(drift + self.sigma * W)
        return traj

    def calibrate(self, trajectory, Dt):
        # TODO: Your code goes here
        if not isinstance(trajectory, list):
            trajectory = list(trajectory)
            
        increments = np.diff(np.log(trajectory))
        
        # Boostrap calculation
        # moments: [E(x), E(x^2)]
        # --> mean = E(x); variance = E(x^2) - E(x)^2
        moments = [0.0, 0.0]
        n_iter = 10
        for iter in range(n_iter):
            X = self.rng.choice(increments, size=len(increments)//2, replace=False)
            Xsq = X**2
            moments[0] += np.mean(X)/n_iter
            moments[1] += np.mean(Xsq)/n_iter
        std = np.sqrt(moments[1] - moments[0]**2)
        self.sigma = std / np.sqrt(Dt)
        self.mu = moments[0] / Dt + self.sigma**2 / 2
        
        
    def forecast(self, latest, T, confidence):
        # TODO: Your code goes here
        predicted = latest * np.exp(self.mu * T)
        m = (self.mu - self.sigma**2 / 2) * T
        s = self.sigma * np.sqrt(T)
        lower_bound = norm.ppf((1-confidence)/2, loc=m, scale=s)
        upper_bound = norm.ppf((1+confidence)/2, loc=m, scale=s)
        # return dict of results
        if not isinstance(latest, (int, float)):
            latest = float(latest)
        return {'expected': predicted,
                'confidence': confidence,
                'interval': latest * np.exp([lower_bound, upper_bound])}
        
        
    def expected_shortfall(self, T, confidence):
        # TODO: Your code goes here
        m = (self.mu - self.sigma**2/2) * T
        s = self.sigma * np.sqrt(T)
        
        ES = -m + s * norm.pdf(norm.ppf(confidence))/(1 - confidence)
        return ES
