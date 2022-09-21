import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import pylab

def make_hist(data):
    #### General code:
    bins_formulas = ['auto', 'fd', 'scott', 'rice', 'sturges', 'doane', 'sqrt']
    bins = np.histogram_bin_edges(a=data, bins='fd', range=(min(data), max(data)))
    # print('Bin value = ', bins)

    # Obtaining the histogram of data:
    Hist, bin_edges = histogram(a=data, bins=bins, range=(min(data), max(data)), density=True)
    bin_mid = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0  # go from bin edges to bin middles
    return Hist, bin_mid


def make_pdf(dist, params, size):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return pdf, x, y


def compute_r2_test(y_true, y_predicted):
    sse = sum((y_true - y_predicted) ** 2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score, sse, tse


def get_best_distribution_2(data, method, plot=False):
    dist_names = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2',
                  'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                  'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon',
                  'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
                  'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss',
                  'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma',
                  'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct',
                  'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal',
                  'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm',
                  'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']

    # Applying the Goodness-to-fit tests to select the best distribution that fits the data:
    dist_results = []
    dist_IC_results = []
    params = {}
    params_IC = {}
    params_SSE = {}

    if method == 'sse':
        ########################################################################################################################
        ######################################## Sum of Square Error (SSE) test ################################################
        ########################################################################################################################
        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param
            N_len = len(list(data))
            # Obtaining the histogram:
            Hist_data, bin_data = make_hist(data=data)

            # fit dist to data
            params_dist = dist.fit(data)

            # Separate parts of parameters
            arg = params_dist[:-2]
            loc = params_dist[-2]
            scale = params_dist[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = dist.pdf(bin_data, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(Hist_data - pdf, 2.0))

            # identify if this distribution is better
            if best_sse > sse > 0:
                best_distribution = dist
                best_params = params_dist
                best_stat_test_val = sse

        print(
            '\n################################ Sum of Square Error test parameters #####################################')
        best_dist = best_distribution
        print("Best fitting distribution (SSE test) :" + str(best_dist))
        print("Best SSE value (SSE test) :" + str(best_stat_test_val))
        print("Parameters for the best fit (SSE test) :" + str(params[best_dist]))
        print(
            '###########################################################################################################\n')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    if method == 'r2':
    ########################################################################################################################
    ##################################################### R Square (R^2) test ##############################################
    ########################################################################################################################
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_r2 = np.inf

    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        N_len = len(list(data))
        # Obtaining the histogram:
        Hist_data, bin_data = make_hist(data=data)

        # fit dist to data
        params_dist = dist.fit(data)

        # Separate parts of parameters
        arg = params_dist[:-2]
        loc = params_dist[-2]
        scale = params_dist[-1]

        # Calculate fitted PDF and error with fit in distribution
        pdf = dist.pdf(bin_data, loc=loc, scale=scale, *arg)
        r2 = compute_r2_test(y_true=Hist_data, y_predicted=pdf)

        # identify if this distribution is better
        if best_r2 > r2 > 0:
            best_distribution = dist
            best_params = params_dist
            best_stat_test_val = r2

    print('\n############################## R Square test parameters ###########################################')
    best_dist = best_distribution
    print("Best fitting distribution (R^2 test) :" + str(best_dist))
    print("Best R^2 value (R^2 test) :" + str(best_stat_test_val))
    print("Parameters for the best fit (R^2 test) :" + str(params[best_dist]))
    print('#####################################################################################################\n')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    if method == 'ic':
        ########################################################################################################################
        ######################################## Information Criteria (IC) test ################################################
        ########################################################################################################################
        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param
            N_len = len(list(data))

            # Obtaining the histogram:
            Hist_data, bin_data = make_hist(data=data)

            # fit dist to data
            params_dist = dist.fit(data)

            # Separate parts of parameters
            arg = params_dist[:-2]
            loc = params_dist[-2]
            scale = params_dist[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = dist.pdf(bin_data, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(Hist_data - pdf, 2.0))

            # Obtaining the log of the pdf:
            loglik = np.sum(dist.logpdf(bin_data, *params_dist))
            k = len(params_dist[:])
            n = len(data)
            aic = 2 * k - 2 * loglik
            bic = n * np.log(sse / n) + k * np.log(n)
            dist_IC_results.append((dist_name, aic))
            # dist_IC_results.append((dist_name, bic))

        # select the best fitted distribution and store the name of the best fit and its IC value
        best_dist, best_ic = (min(dist_IC_results, key=lambda item: item[1]))

        print(
            '\n############################ Information Criteria (IC) test parameters ##################################')
        print("Best fitting distribution (IC test) :" + str(best_dist))
        print("Best IC value (IC test) :" + str(best_ic))
        print("Parameters for the best fit (IC test) :" + str(params[best_dist]))
        print(
            '###########################################################################################################\n')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    if method == 'chi':
        ########################################################################################################################
        ################################################ Chi-Square (Chi^2) test ###############################################
        ########################################################################################################################
        # Set up 50 bins for chi-square test
        # Observed data will be approximately evenly distrubuted aross all bins
        percentile_bins = np.linspace(0, 100, 51)
        percentile_cutoffs = np.percentile(data, percentile_bins)
        observed_frequency, bins = (np.histogram(data, bins=percentile_cutoffs))
        cum_observed_frequency = np.cumsum(observed_frequency)

        chi_square = []
        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param

            # Obtaining the histogram:
            Hist_data, bin_data = make_hist(data=data)

            # fit dist to data
            params_dist = dist.fit(data)

            # Separate parts of parameters
            arg = params_dist[:-2]
            loc = params_dist[-2]
            scale = params_dist[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = dist.pdf(bin_data, loc=loc, scale=scale, *arg)

            # Get expected counts in percentile bins
            # This is based on a 'cumulative distrubution function' (cdf)
            cdf_fitted = dist.cdf(percentile_cutoffs, *arg, loc=loc, scale=scale)
            expected_frequency = []
            for bin in range(len(percentile_bins) - 1):
                expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                expected_frequency.append(expected_cdf_area)

            # calculate chi-squared
            expected_frequency = np.array(expected_frequency) * size
            cum_expected_frequency = np.cumsum(expected_frequency)
            ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
            chi_square.append(ss)

            # Applying the Chi-Square test:
            # D, p = scipy.stats.chisquare(f_obs=pdf, f_exp=Hist_data)
            # print("Chi-Square test Statistics value for " + dist_name + " = " + str(D))
            print("p value for " + dist_name + " = " + str(chi_square))
            dist_results.append((dist_name, chi_square))

        # select the best fitted distribution and store the name of the best fit and its p value
        best_dist, best_stat_test_val = (min(dist_results, key=lambda item: item[1]))

        print(
            '\n#################################### Chi-Square test parameters #######################################')
        print("Best fitting distribution (Chi^2 test) :" + str(best_dist))
        print("Best p value (Chi^2 test) :" + str(best_stat_test_val))
        print("Parameters for the best fit (Chi^2 test) :" + str(params[best_dist]))
        print(
            '#########################################################################################################\n')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    if method == 'ks':
        ########################################################################################################################
        ########################################## Kolmogorov-Smirnov (KS) test ################################################
        ########################################################################################################################
        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param

            # Applying the Kolmogorov-Smirnov test:
            D, p = st.kstest(data, dist_name, args=param)
            # D, p = st.kstest(data, dist_name, args=param, N=N_len, alternative='greater')
            # print("Kolmogorov-Smirnov test Statistics value for " + dist_name + " = " + str(D))
            print("p value for " + dist_name + " = " + str(p))
            dist_results.append((dist_name, p))

        # select the best fitted distribution and store the name of the best fit and its p value
        best_dist, best_stat_test_val = (max(dist_results, key=lambda item: item[1]))

        print(
            '\n################################ Kolmogorov-Smirnov test parameters #####################################')
        print("Best fitting distribution (KS test) :" + str(best_dist))
        print("Best p value (KS test) :" + str(best_stat_test_val))
        print("Parameters for the best fit (KS test) :" + str(params[best_dist]))
        print(
            '###########################################################################################################\n')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    # Collate results and sort by goodness of fit (best at top)
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    # results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)

    # Plotting the distribution with histogram:
    if plot:
        bins_val = np.histogram_bin_edges(a=data, bins='fd', range=(min(data), max(data)))
        plt.hist(x=data, bins=bins_val, range=(min(data), max(data)), density=True)
        # pylab.hist(x=data, bins=bins_val, range=(min(data), max(data)))
        best_param = params[best_dist]
        best_dist_p = getattr(st, best_dist)
        pdf, x_axis_pdf, y_axis_pdf = make_pdf(dist=best_dist_p, params=best_param, size=len(data))
        plt.plot(x_axis_pdf, y_axis_pdf, color='red', label='Best dist ={0}'.format(best_dist))
        plt.legend()
        plt.title('Histogram and Distribution plot of data')
        # plt.show()
        plt.show(block=False)
        plt.pause(5)  # Pauses the program for 5 seconds
        plt.close('all')

    return best_dist, best_stat_test_val, params[best_dist]