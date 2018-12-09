import numpy as np


def ornstein_uhlenbeck_noise(arr, theta=.15, sig=.2, mu=0.8):
    dt = np.mean(np.diff(arr))

    y = np.zeros_like(arr)
    y0 = np.random.normal()

    drift = lambda y, arr: theta*(mu -y)
    diffusion = lambda y,t: sigma
    noise = np.random.normal(loc=0.0, scale=1.0, size=len(arr))*np.sqrt(dt)

    print(drift, diffusion, noise, y, y0, dt)
    
arr = np.random.rand(2,3)

ornstein_uhlenbeck_noise(arr)
