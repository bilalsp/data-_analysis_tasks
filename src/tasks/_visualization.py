import os
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_data

plt.style.use('ggplot')


class Visualization:

    def __init__(self) -> None:
        """Produce Visualization"""
        self.output_path = 'outputs/visuals/'
        self.data_path = 'data/pop-sexe-age-quinquennal6817.xls'
        self.sheet = 'COM_2017'
        self.skip_rows = 14 # header rows
        self.age_groups = ["{}-{}".format(i,i+4) for i in range(0, 91, 5)] 
        self.age_groups.append('90+')
      
    def run(self, *args, verbose=True) -> None:
        """Run Visualization task"""
        text_data, _ = load_data(
            self.data_path, self.sheet, self.skip_rows, verbose=verbose)
        data = np.genfromtxt(StringIO(text_data), delimiter=',', dtype=object)
        data = np.where(data==b'', b'0', data) # replace empty string with b'0'
    
        # Population Info --> Column index 6 to 46 represents population
        population = data[:, range(6, 46)].astype(np.float32)
 
        # population data has floating point. But population can't be floating 
        # number. So, we round it to integer 
        population = np.rint(population)

        # 1. an age pyramid for France in 2017
        self._plot_age_pyramid(population, filename='age_pyramid')

        # 2. frequency histogram of the number of inhabitants, across all 
        # municipalities (total all ages)
        population = population.sum(axis=1) # population of each municipality
        self._plot_hist(
            population, 
            title='Frequency histogram of the number of inhabitants',
            filename='inhabitants_histogram')

        # Observation: Frequency histogram is not properly visible because of
        # very high difference between quartiles Q3 and Q4 
        # To properly visualze the histogram, we use the data below 
        # 95-percentile

        p95 = np.percentile(population, [95])
        population = population[population <= p95]
        self._plot_hist(
            population, 
            title="""Frequency histogram of the number of inhabitants which 
                     are below 95-percentile""",
            filename='inhabitants_histogram_95p')
        
        # Observation: Densely populated municipalities are less in number than
        # sparsely populated municipalities

        if verbose:
            print('Saved visuals in {} directory'.format(self.output_path))

    def _plot_age_pyramid(self, population, filename=None):
        # male and female group population (Alternative columns in population)
        male_grp_population = population[:, range(0, population.shape[1], 2)]
        male_grp_population = male_grp_population.sum(axis=0)
        female_grp_population = population[:, range(1, population.shape[1], 2)]
        female_grp_population = female_grp_population.sum(axis=0)

        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
        fig.suptitle('An age pyramid for France in 2017', 
                     fontweight ="bold", fontsize=16)
    
        y = range(len(male_grp_population))
        axs[0].barh(y, male_grp_population, align='center', color='royalblue')
        axs[0].set_title('Males', fontsize=12)
        axs[0].set_xlabel('Male Population', fontsize=8) 
        axs[0].set_ylabel('Age Group', fontsize=8)
        axs[1].barh(y, female_grp_population, align='center', color='lightpink')
        axs[1].set_title('Females', fontsize=12)
        axs[1].set_xlabel('Female Population', fontsize=8) 
        
        # adjust grid parameters and specify labels for y-axis
        axs[1].grid()
        axs[0].grid()
        axs[0].set(yticks=y, yticklabels=self.age_groups)
        axs[0].invert_xaxis()
        
        plt.subplots_adjust(wspace=0, hspace=0)

        if filename:
            os.makedirs(self.output_path, exist_ok=True)
            fig.savefig(self.output_path+filename, dpi=fig.dpi)

    def _plot_hist(self, population, title, filename=None):
        # number of bins:  `Freedman–Diaconis` rule
        q75, q25 = np.percentile(population, [75 ,25])
        iqr = q75 - q25 # interquartile range
        bin_width = 2 * iqr * (len(population)**(-1/3))
        bins = (np.max(population) - np.min(population))/bin_width
        bins = int(bins)

        fig, ax = plt.subplots(figsize=(10, 6))    
        n, bins, patches = ax.hist(population, bins, density=False, 
                                   facecolor='g', alpha=0.75)
        ax.set_xlabel('Total Population of municipality', fontsize=8) 
        ax.set_ylabel('Frequency', fontsize=8)
        
        quartiles = np.percentile(population, [25, 50, 75, 100])
        q_text = "\n".join( ['Quartiles: '] + 
            ['Q{}: {}'.format(i+1, val) for i, val in enumerate(quartiles)])
        ax.text(0.8, 0.90, q_text, 
                color='darkblue', fontsize=8, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

        fig.suptitle(title, fontweight ="bold", fontsize=16)
    
        if filename:
            os.makedirs(self.output_path, exist_ok=True)
            fig.savefig(self.output_path+filename, dpi=fig.dpi)
            
