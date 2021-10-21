import os
from io import StringIO

import numpy as np
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt

from src.utils import load_data
from src.tasks._univariate_analysis import UnivariateAnalysis

plt.style.use('ggplot')


class BivariateAnalysis :

    def __init__(self) -> None:
        """Bivariate analysis between percentage of 15/24 year olds and 
        the median of declared income"""
        self.output_path = 'outputs/visuals/'
        self.data_path = 'data/FILO2018_DEC_COM.xls'
        self.sheet = 'ENSEMBLE'
        self.skip_rows = 6 # header rows
      
    def run(self, population_perct_15_24_age, verbose=True) -> None:
        """Run bivariate analysis
        
        Args:
            population_perct_15_24_age: (optional) population percentage of 
            15-24 age group
        """
        # This task has dependency on UnivariateAnalysis for
        #  `population percentage of 15-24 age group`
        if population_perct_15_24_age is None:
            population_perct_15_24_age = UnivariateAnalysis().run(verbose)   
                 
        text_data, _ = load_data(self.data_path, self.sheet, 
                                 self.skip_rows, verbose=verbose)

        # structured-array consisting of `insee code` and `median salary`
        salary_data = np.genfromtxt(
            StringIO(text_data), delimiter=',', usecols=[0,7], 
            dtype=[('insee_code', '<U20'), ('median_salary', np.float32)])

        # structured-array consisting of `insee_code` and `population percentage 
        # of 15-25 age group`
        population_data = rfn.unstructured_to_structured(
            population_perct_15_24_age, 
            dtype=np.dtype(
                [('insee_code', '<U20'), ('population_perct', np.float32)]))

        # Join population_data and salary_data on `insee_code` key
        population_salary_data = rfn.join_by(
            'insee_code', population_data, salary_data, jointype='inner', 
            usemask=False)

        # Independent and Dependent Variables in our bivariate analysis
        # Independent Variable: population_percentage of 15-24 age group
        # Dependent Variable: median of declared salary
        population_perct = population_salary_data['population_perct']
        median_salary = population_salary_data['median_salary']

        # Scatter Plot
        self._scatter_plot(
            x=population_perct, 
            y=median_salary, 
            title="Bivariate analysis",
            filename='bivariate_analysis')
        # Observation: Data points are widely scattered, forming a cloud of 
        # points with some extreme outliers. We also observe the lack of 
        # Homoscedasticity (equal variability) in the data.
        # This scatter plot represents high number of data points (31243).
        # That's why we might be observing overplotting (because 
        # of data overlap or dense data). 
        # To further analysis this interpretation, we draw scatter plot for
        # random sample (i.e., 2% data points)
        # Furthermore, person's r correlation coefficient is very low which 
        # express week correlation

        np.random.seed(12345)
        size = int(population_perct.shape[0]*0.02)
        random_index = np.random.randint(0, population_perct.shape[0], size)
        self._scatter_plot(
            x=population_perct[random_index], 
            y=median_salary[random_index], 
            title="Bivariate analysis on subset (2% data points)",
            filename='bivariate_analysis_subset')

        # Observation: We still observe heteroscedasticity in the data and low
        # correlation on random samples. That is a voilatation of the OLS (ordinary
        # least sequares) assumption. 

        if verbose:
            print('Saved Bivariate Analysis Visual in {} directory'.format(
                self.output_path))

    def _scatter_plot(self, x, y, title, filename=None):
        fig, ax = plt.subplots()   
        ax.scatter(x, y, s=1, alpha=0.5, linewidths=1)
        ax.set_title(title, fontweight ="bold", fontsize=16)
        ax.set_xlabel('Percentage of 15-24 year age olds (x)', fontsize=8)      
        ax.set_ylabel('Median of declared income (y)', fontsize=8)      

        # Correlation Coefficient (Pearson's r)
        corr = np.round(np.corrcoef(x, y)[0,1], 3)
        ax.text(0.75, 0.95, 'Correlation: {}'.format(corr), 
                color='darkblue', fontsize=8, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

        # Least squares regression fit
        regression_coeff = np.polyfit(x, y, 1)
        ax.plot(x, regression_coeff[0] * x + regression_coeff[1], 
                color='darkblue', linewidth=1)
   
        # regression equation
        ax.text(0.75, 0.9, 'Regression Line: y={:.2f}*x+{:.2f}'.format(
            regression_coeff[0], regression_coeff[1]), color='darkblue', 
            fontsize=8, horizontalalignment='center', 
            verticalalignment='center', transform=ax.transAxes)

        if filename:
            os.makedirs(self.output_path, exist_ok=True)
            fig.savefig(self.output_path+filename, dpi=fig.dpi)
    