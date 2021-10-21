from io import StringIO
import numpy as np
from src.utils import load_data


class UnivariateAnalysis:

    def __init__(self) -> None:
        """Univariate Analysis of 15-24 age group"""
        self.output_path = 'outputs/'
        self.data_path = 'data/pop-sexe-age-quinquennal6817.xls'
        self.sheet = 'COM_2017'
        self.skip_rows = 14 # header rows
        # columns name in dataset which represent 15-24 age group
        self.age_15_24_cols = ['ageq_rec04s1rpop2017', 'ageq_rec04s2rpop2017',
                               'ageq_rec05s1rpop2017', 'ageq_rec05s2rpop2017']

    def run(self, *args, verbose=True) -> np.ndarray:
        text_data, desc = load_data(self.data_path, self.sheet, 
                                    skip_rows=self.skip_rows)
        data = np.genfromtxt(StringIO(text_data), delimiter=',', dtype=object)
        data = np.where(data==b'', b'0', data) # replace empty string with b'0'
        data_col = np.array(desc['column_name'])  
        
        # Region Info --> 'DR', 'CR', 'LIBELLE'
        region = np.char.decode(
            data[:, np.isin(data_col, ['DR', 'CR', 'LIBELLE'])]\
                .astype(np.string_), encoding='ISO-8859-1')
        
        # municipality_id or insee code = DR + CR
        municipality_ids = np.char.add(region[:,0], region[:,1])
        
        # add new column (municipality_id) into Region Info
        region = np.hstack((region, municipality_ids[:, np.newaxis]))

        # Population Info --> Column index 6 to 46 represents population
        population = data[:, range(6, 46)].astype(np.float32)
        population_col = data_col[range(6, 46)]

        # population data has floating point. But population can't be floating 
        # number. So, we round it to integer 
        population = np.rint(population)

        # Population of 15-24 year olds age group for each municipality
        population_15_24_age_col = np.array(self.age_15_24_cols)
        population_15_24_age = population[:, np.isin(population_col, 
                                                     population_15_24_age_col)]
        population_15_24_age = population_15_24_age.sum(axis=1, keepdims=True)


        ###########################Sub-Tasks################################
        population_perct_15_24_age = self._cal_perct_age_group_15_24(
            population, population_15_24_age)

        avg_number_15_24_age, std_15_24_age = self._cal_avg_no_age_group_15_24(
            population_15_24_age)

        avg_perct_15_24 = self._cal_avg_perct_age_group_15_24(
            population, population_15_24_age)

        extreme_municipalities = self._get_extreme_municipalities( 
            population, population_perct_15_24_age, region)

        # population percentage of 15-24 age group with INSEE code (DR+CR)
        population_perct_15_24_age = np.hstack(
            (region[:, 3:], population_perct_15_24_age))

        if verbose:
            print('Average number of 15/24 and the standard deviation', end=':')
            print(avg_number_15_24_age, std_15_24_age)
            print('Average percentage of 15/24 in France', end=': ')
            print(avg_perct_15_24)
            print("""Municipalities with an extreme value of the percentage of 15/24""")
            print(extreme_municipalities)
        
        with open(self.output_path+'population_perct_15_24_age.npy', 'wb') as f:
            np.save(f, population_perct_15_24_age)

        return population_perct_15_24_age

    def _cal_perct_age_group_15_24(self, population, population_15_24_age):
        """1. calculate the percentage of 15/24 year olds for each municipality"""
        population_of_municipality = population.sum(axis=1, keepdims=True)
        population_perct_15_24_age = np.divide(population_15_24_age, 
            population_of_municipality, where=population_of_municipality!=0.0,
            out=np.zeros_like(population_15_24_age)) * 100
        return population_perct_15_24_age

    def _cal_avg_no_age_group_15_24(self, population_15_24_age):
        """2. display the average number of 15/24 and the standard deviation"""
        avg_number_15_24_age = np.mean(population_15_24_age)
        std_15_24_age = np.std(population_15_24_age)
        return avg_number_15_24_age, std_15_24_age

    def _cal_avg_perct_age_group_15_24(self, population, population_15_24_age):
        """3. display the average percentage of 15/24 in France"""
        avg_perct_15_24 = np.divide(population_15_24_age.sum(), 
                                    population.sum()) * 100
        return avg_perct_15_24

    def _get_extreme_municipalities(self, 
                                    population, 
                                    population_perct_15_24_age,
                                    region):
        """4. find the municipalities with an extreme value of the percentage of 
        15/24 (extreme high and low) and indicate their name, insee code, 
        percentage 15/24, population"""
              
        def get_extreme_low_idx(nd_array, mask):
            # return extreme low index based on masking
            subset_idx = np.argmin(nd_array[mask])
            extreme_low_idx = np.expand_dims(
                np.arange(nd_array.shape[0]), axis=-1)[mask][subset_idx]
            return extreme_low_idx

        # consider only those municipalities which has population > 0
        population_of_municipality = population.sum(axis=1, keepdims=True) 
        mask = population_of_municipality > 0.0
        extreme_low_idx = get_extreme_low_idx(population_perct_15_24_age, mask)
        extreme_high_idx = np.argmax(population_perct_15_24_age)

        # extreme Low
        extreme_municipalities = [(
            region[extreme_low_idx, 2],  # name
            region[extreme_low_idx, 3],# insee code
            population_perct_15_24_age[extreme_low_idx].item(),#percentage 15/24
            population_of_municipality[extreme_low_idx].item(), #population
            'low'
        )]

        # extreme High
        extreme_municipalities.append((
            region[extreme_high_idx, 2],
            region[extreme_high_idx, 0] + region[extreme_high_idx, 1], 
            population_perct_15_24_age[extreme_high_idx].item(),
            population_of_municipality[extreme_high_idx].item(),
            'high'
        ))

        return extreme_municipalities
