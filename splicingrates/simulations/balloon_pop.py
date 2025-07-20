import marimo

__generated_with = "0.6.0"
app = marimo.App()


@app.cell
def __():
    # let's show it for the case that N=10, h=5
    import numpy as np
    import itertools
    N=20
    max_time = 10
    t_i = np.random.uniform(0, max_time, N)
    h = 5
    p_i = 2**(-t_i/h)
    # first, calculate the probability that there are n balls left at 12PM
    n = np.arange(0, N+1)
    p_n = np.zeros(N+1)
    for i in range(N+1):
        # get all the subset of size i of {0,...N}
        for subset in itertools.combinations(range(N), i):
            p_n[i] += np.prod(p_i[list(subset)])*np.prod(1-p_i[list(set(range(N))-set(subset))])

    # second, let s ~ Poission(mean = \sum_{i=1}^N p_i)
    # calculate the probability of s=n for n=0,...,N
    def poisson_probabilities(lam, N=10):
        from scipy.stats import poisson
        # Create an array of possible values of s
        s_values = np.arange(0, N+1)
        # Calculate the Poisson probabilities for each value of s
        probabilities = poisson.pmf(s_values, lam)
        # Create a dictionary to store the probabilities
        prob_dict = {n: prob for n, prob in zip(s_values, probabilities)}
        return prob_dict

    def normal_probabilities(lam, var, N=10):
        from scipy.stats import norm
        # Create an array of possible values of s
        s_values = np.arange(0, N+1)
        # Calculate the Poisson probabilities for each value of s
        probabilities = norm.pdf(s_values, lam, np.sqrt(var))
        # Create a dictionary to store the probabilities
        prob_dict = {n: prob for n, prob in zip(s_values, probabilities)}
        return prob_dict

    lam = np.sum(p_i)
    one_minus_p_i = 1-p_i
    # variance of the normal distribution
    var = np.sum(p_i*one_minus_p_i)
    poisson_prob = poisson_probabilities(lam, N)
    normal_prob = normal_probabilities(lam, var, N)
    # now, draw the p_n and the poisson_prob on the same plot
    import matplotlib.pyplot as plt
    plt.plot(n, p_n, 'r', label='p_n')
    plt.plot(n, [poisson_prob[i] for i in n], 'b', label='poisson_prob')
    plt.plot(n, [normal_prob[i] for i in n], 'g', label='normal_prob')
    plt.legend()
    plt.show()
    return (
        N,
        h,
        i,
        itertools,
        lam,
        max_time,
        n,
        normal_prob,
        normal_probabilities,
        np,
        one_minus_p_i,
        p_i,
        p_n,
        plt,
        poisson_prob,
        poisson_probabilities,
        subset,
        t_i,
        var,
    )


@app.cell
def __(N, h, np, plt, t_i):
    # now, given the fixed t_i, N. We will generate n multiple times (like, doing the ball poppoing experiment multiple times) and draw the distribution of n
    def do_ballon_pop_exp_multiple_times(N, t_i, h, num_times=1000):
        p_i = 2**(-t_i/h)
        n = np.zeros(num_times)
        for i in range(num_times):
            x_i = np.random.uniform(0, 1, N)
            n[i] = np.sum(x_i < p_i)
        return n


    _n = do_ballon_pop_exp_multiple_times(N=N, t_i = t_i, h=h, num_times=10000)
    plt.hist(_n, bins=100, density=False)
    plt.show()
    return do_ballon_pop_exp_multiple_times,


@app.cell
def __(N, do_ballon_pop_exp_multiple_times, h, np, plt, t_i):
    # gievn each of these n, can we estimate h using optimization method
    from estimate_splice import optimize_h, find_root
    _n = do_ballon_pop_exp_multiple_times(N=N, t_i = t_i, h=h, num_times=1000)
    plt.hist(_n, bins=100, density=False)
    plt.show()
    h_hat_opt = np.zeros(len(_n))
    h_hat_root = np.zeros(len(_n))
    for j in range(len(_n)):
        h_hat_opt[j]= optimize_h([t_i], [_n[j]])
        h_hat_root[j] = find_root([t_i], [_n[j]])

    # in two plots next to each other, plot the histogram of the estimated h_hat_opt and h_hat_root
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(h_hat_opt, bins=100, density=False)
    axs[1].hist(h_hat_root, bins=100, density=False)
    plt.show()
    return axs, fig, find_root, h_hat_opt, h_hat_root, j, optimize_h


@app.cell
def __(h_hat_opt, h_hat_root, np):
    print(np.mean(h_hat_opt))
    print(np.mean(h_hat_root))
    return


@app.cell
def __(optimize_h, t_i):
    t = optimize_h([t_i], [11])
    print(t)
    return t,


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
