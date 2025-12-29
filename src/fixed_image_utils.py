from sklearn.mixture import GaussianMixture
import pandas as pd
from pathlib import Path
import re

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from live_image_utils import extract_wells_info

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Function to fit the Gaussian to your data
def fit_gaussian_to_noninfected(non_infected_values, bins=100):
    # Create histogram
    counts, bin_edges = np.histogram(non_infected_values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial parameter guess [amplitude, mean, standard_deviation]
    # For the initial guess, we can use some simple statistics
    p0 = [np.max(counts), np.mean(non_infected_values), np.std(non_infected_values)]
    
    # Fit the Gaussian
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=p0)
        return popt, pcov, bin_centers, counts
    except RuntimeError:
        print("Error - curve_fit failed")
        
        return p0, None, bin_centers, counts
# Define a two-component Gaussian mixture model with proper weights
def gaussian_mixture(x, weight, mu1, sigma1, mu2, sigma2):
    """
    Two-component Gaussian mixture model with proper normalization.
    - weight: proportion of first component (0 to 1)
    - First component (mu1, sigma1) represents non-infected cells
    - Second component (mu2, sigma2) represents infected cells
    
    Note: Each component integrates to 1, and weight controls their proportion
    """
    comp1 = (1/np.sqrt(2*np.pi*sigma1**2)) * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    comp2 = (1/np.sqrt(2*np.pi*sigma2**2)) * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return weight * comp1 + (1 - weight) * comp2

def fit_gaussian_mixture_to_infected_wells(infected_values, non_infected_params, bins=100):
    """
    Fit a two-component Gaussian mixture to infected well data,
    constraining the first component parameters (mean and std) using parameters from non-infected cells,
    but letting the weight (proportion) be determined by the fit.
    
    Parameters:
    - infected_values: array of viral signal values from infected wells
    - non_infected_params: [amp, mu1, sigma1] parameters from non-infected Gaussian fit
    - bins: number of bins for histogram
    
    Returns:
    - popt: Optimized parameters for the mixture [weight, mu1, sigma1, mu2, sigma2]
    - pcov: Covariance matrix for parameter optimization
    - bin_centers, counts: Histogram data
    """
    # Create histogram of infected well data
    counts, bin_edges = np.histogram(infected_values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Extract mean and standard deviation from non-infected fit
    _, mu1, sigma1 = non_infected_params
    
    # Define a constrained version of the mixture model where mu1 and sigma1 are fixed
    def constrained_mixture(x, weight, mu2, sigma2):
        return gaussian_mixture(x, weight, mu1, sigma1, mu2, sigma2)
    
    # Initial guess for parameters
    # Start with guessing 50% non-infected cells
    weight_initial = 0.5  
    mu2_initial = mu1 * 2  # Guess that infected mean is twice non-infected mean
    sigma2_initial = sigma1 * 1.5  # Guess that infected stdev is 1.5x non-infected stdev
    
    p0 = [weight_initial, mu2_initial, sigma2_initial]
    
    # Bounds to ensure reasonable parameters
    # (weight, mean, stdev)
    lower_bounds = [0.01, mu1, sigma1/2]  # Weight > 0, mu2 > mu1, sigma2 > 0
    upper_bounds = [0.99, np.max(infected_values) * 1.2, sigma1 * 5]
    
    # Fit the mixture model with constrained component parameters
    try:
        popt, pcov = curve_fit(constrained_mixture, bin_centers, counts, 
                              p0=p0, bounds=(lower_bounds, upper_bounds))
        
        # Full parameters including fixed parameters
        full_params = [popt[0], mu1, sigma1, popt[1], popt[2]]
        
        return full_params, pcov, bin_centers, counts
    except RuntimeError as e:
        print(f"Error fitting mixture model: {e}")
        return None, None, bin_centers, counts

def analytical_intersection(w, m1, s1, m2, s2):
    d = s1**2-s2**2
    b = 2*m1*s2**2 - 2*m2*s1**2
    delta = b**2 - 4*d*(2*s1**2*s2**2*np.log(s2*w/s1/(1-w))+(m2**2*s1**2-m1**2*s2**2))
    print("delta: ", delta)
    if delta>0:
        x1 = (-b-np.sqrt(delta))/d/2
        print(x1, (-b+np.sqrt(delta))/d/2)
        if np.min((m1, m2))<x1 and x1<np.max((m1, m2)):
            return x1
        else:
            return (-b+np.sqrt(delta))/d/2
    return None

# Function to calculate proportion of infected cells per group
def calculate_infection_proportion(df):
    # First check if 'Infected' column is boolean and convert if needed
    #df = df[df['']]
    if 'Infected' not in df.columns:
        df['Infected'] = df['virus_intensity_sum']>df['threshold']
    if df['Infected'].dtype == 'object':
        df['Infected'] = df['Infected'].map({'True': True, 'False': False})
    
    # Group by the required columns and calculate the proportion
    grouped = df.groupby(['experiment', 'CellType1', 'CellType2', 'Virus'])
    
    # Calculate the proportion of infected cells in each group
    result = grouped.agg(
        infected_count=('Infected', lambda x: sum(x == True)),
        total_count=('Infected', 'count')
    )
    
    # Calculate the proportion
    result['proportion_infected'] = result['infected_count'] / result['total_count']
    
    # Reset index to convert back to regular dataframe
    result = result.reset_index()
    
    return result

def read_data(paths, use_mock=False):
    nuclei = []
    col = "virus_intensity_mean"
    for path in paths:
        exp_nuclei = []
        print("Read: ", path)
        for well in Path(path).glob('./*-ST/'):
            nuc_quant = pd.read_csv(well/'Img_t0000-nuc_quant.csv')
            img_quant = pd.read_csv(well/'Img_t0000-img_quant.csv')

            #nuc_quant = nuc_quant.merge(img_quant, how="left", suffixes=("", "_img"))
            for col_name in img_quant.columns:
                nuc_quant[col_name + "_img"] = img_quant.loc[0, col_name]
            nuc_quant["experiment"] = path.split("/")[-1]
            nuc_quant["well"] = re.search(r'(\d+)-ST', str(well)).group(1)
            exp_nuclei.append(nuc_quant)
        if len(exp_nuclei)==0:
            continue
        exp_nuclei = pd.concat(exp_nuclei)
        info = extract_wells_info(path)

        exp = exp_nuclei.merge(info, on=['well', 'experiment']).rename({"iav_intensity_mean": "virus_intensity_mean", "rsv_intensity_mean": "virus_intensity_mean"}, axis=1)
        if 'virus_intensity_mean' not in exp.columns:
            exp['virus_intensity_mean'] = exp['virus2_intensity_mean']
        if len(exp[exp["Virus"]=='mock'])!=0 and use_mock:
            popt, pcov, bin_centers, counts = fit_gaussian_to_noninfected(exp[col][exp["Virus"]=='mock'])
            if pcov is not None:
                infected_params, icov, bin_centers, counts = fit_gaussian_mixture_to_infected_wells(exp[col], popt)#[exp["Virus"]!='mock']
                print(infected_params)
                threshold = analytical_intersection(*infected_params)
                print(threshold, " from double-fit")
            else: 
                threshold = popt[0]+2*popt[1]

        else:
            gmm = GaussianMixture(n_components=2, covariance_type='full', init_params='kmeans', random_state=42)
            gmm.fit(exp[col].values.reshape(-1, 1))

            # Identify which component matches the mock Gaussian
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            # Extract proportions
            proportions = gmm.weights_
            neg_idx = np.argmax(means)
            print(means, variances, neg_idx)
            threshold = analytical_intersection(proportions[neg_idx], means[neg_idx], np.sqrt(variances[neg_idx]), means[1-neg_idx], np.sqrt(variances[1-neg_idx]))
            print(threshold, " from single-fit")
        exp[exp['Virus']!='mock'][col].hist(bins=255)
        exp[exp['Virus']=='mock'][col].hist(bins=255)
        if threshold is not None:
            plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.show()
        exp['Infected'] = exp[col]>threshold
        exp['threshold'] = threshold
        nuclei.append(exp)

    return pd.concat(nuclei)