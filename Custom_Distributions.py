import numpy as np

def custom_lognorm_rvs(mu, sigma, size=1):
    """
    Fast custom lognormal generator using Box-Muller transform (vectorized).
    """
    # Uniform(0,1) arrays
    u1 = np.random.random(size)
    u2 = np.random.random(size)

    # Avoid log(0)
    u1[u1 == 0] = 1e-10

    # Box-Muller for standard normal
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)

    # Lognormal transformation
    return np.exp(mu + sigma * z)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = custom_lognorm_rvs(np.log(5000), 1.2, size=100000)
    plt.hist(data, bins=100, edgecolor='k')
    plt.title("Custom Lognormal Distribution")
    plt.show()


