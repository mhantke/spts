import numpy as np
from scipy.special import erf

fraction_of_gaussian_area = lambda sigma, x1, x2: 0.5 * ( erf(-x1/(np.sqrt(2)*sigma)) + erf(x2/(np.sqrt(2)*sigma)) )

def test_fraction_of_gaussian_area():
    # Test fraction_of_gaussian_area
    x = np.linspace(-0.5, 0.5, 100000)
    gaussian= lambda sigma: np.exp(-x**2/2./sigma**2)/sigma/np.sqrt(2*np.pi)
    sigma = 0.1
    y = gaussian(sigma)
    x1 = -0.05
    x2 = 0.07
    A_anal = fraction_of_gaussian_area(sigma, x1, x2)
    A_num = y[(x>=x1)*(x<x2)].sum()/x.size
    assert np.isclose(A_anal, A_num)
