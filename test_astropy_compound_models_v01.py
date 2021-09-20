import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from astropy.modeling.physical_models import Drude1D
from astropy.modeling import Fittable1DModel
from astropy.modeling import Parameter
from astropy.modeling.models import custom_model


class BlackBody1D(Fittable1DModel):
    """
    Current astropy BlackBody1D does not play well with Lorentz1D and Gauss1D
    maybe, need to check again, possibly a units issue
    """

    amplitude = Parameter()
    temperature = Parameter()

    @staticmethod
    def evaluate(x, amplitude, temperature):
        """
        """
        return (
            amplitude
            * ((9.7 / x) ** 2)
            * 3.97289e13
            / x ** 3
            / (np.exp(1.4387752e4 / x / temperature) - 1.0)
        )


@custom_model
def blackbody_test(x, amplitude=None, temperature=None):
    """
    Test blackbody custom_model
    """
    return (
        amplitude
        * ((9.7 / x) ** 2)
        * 3.97289e13
        / x ** 3
        / (np.exp(1.4387752e4 / x / temperature) - 1.0)
    )


def main(prof='Gauss', plot=False):
    # Generate fake data
    np.random.seed(0)
    x = np.linspace(-5., 5., 200)
    y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
    y += np.random.normal(0., 0.2, x.shape)

    if prof == 'Gauss':
        # Fit the data using a Gaussian profile
        g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
        fit_g = fitting.LevMarLSQFitter(calc_uncertainties=True)
        g = fit_g(g_init, x, y)
    elif prof == 'Drude':
        # Fit the data using a Drude profile
        g_init = models.Drude1D(amplitude=1., x_0=1., fwhm=1.)
        fit_g = fitting.LevMarLSQFitter(calc_uncertainties=True)
        g = fit_g(g_init, x, y)
    print(g)

    # Define line model
    line = models.Linear1D(slope=0.0001)

    # Define blackbody model
    bb = BlackBody1D(amplitude=1e-24, temperature=1e-10)

    # Blackbody custom
    bbt = blackbody_test(amplitude=1, temperature=1)

    # Combine models

    # Gaussian + line
    comp1 = g_init + line
    fit_m1 = fitting.LevMarLSQFitter(calc_uncertainties=True)
    comp_fit1 = fit_m1(comp1, x, y)

    # Gaussian + bb
    comp2 = g_init + bb 
    fit_m2 = fitting.LevMarLSQFitter(calc_uncertainties=True)
    comp_fit2 = fit_m2(comp2, x, y)

    # Gaussian + bbt
    comp3 = g_init + bbt
    fit_m3 = fitting.LevMarLSQFitter(calc_uncertainties=True)
    comp_fit3 = fit_m3(comp3, x, y)

    print(comp_fit1, comp_fit2, comp_fit3)
    print(f'\ncomp_fit1 amplitude_0 std: {comp_fit1.amplitude_0.std}')
    print(f'comp_fit2 amplitude_0 std: {comp_fit2.amplitude_0.std}')
    print(f'comp_fit3 amplitude_0 std: {comp_fit3.amplitude_0.std}\n')

    if plot:
        # Plot the data with the best-fit model
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'ko')
        plt.plot(x, g(x), label=f'{prof}')
        plt.plot(x, comp_fit1(x), label=f'{prof}+line')
        plt.plot(x, comp_fit2(x), label=f'{prof}+bb')
        plt.plot(x, comp_fit3(x), label=f'{prof}+bbt')
        plt.xlabel('Position')
        plt.ylabel('Flux')
        plt.legend(loc=2)

        plt.show()


if __name__ == '__main__':
    main(prof='Gauss', plot=False)

