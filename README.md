# Data Analysis and Machine Learning Tasks

<!-- Describe your project in brief -->
This repository contains four basic tasks of Data Analysis and Machine Learning.

# Table of contents
- [Description](#Description)
- [Project Directory Structure](#Project-Directory-Structure)
- [Usage](#Usage)

## Description
There are following four warmup tasks of Data Analysis and Machine Learning in this repository:
Task1: Visualization
Task2: Univariate Analysis
Task3: Bivariate Analysis
Task4: Cluster Analysis
Details description is available in [testdev-ML.txt](testdev-ML.txt).

## Usage
Run an appropriate command inside the project folder. 

To install the dependencies
```
pip install -r requirements.txt
```

To run all the tasks
```
python main.py -tl Visualization,UnivariateAnalysis,BivariateAnalysis,ClusterAnalysis --verbose
```

To run the specific task, pass the appropriate task name  
```
python main.py -tl Visualization --verbose
```

## Datasets
[Population by sex and five-year age from 1968 to 2017 (1990 to 2017 for the DOM)](https://www.insee.fr/fr/statistiques/1893204)

[Income structure and distribution, inequality of living standards in 2018](https://www.insee.fr/fr/statistiques/5009218)

[Customers Data for Cluster Analysis](https://www.kaggle.com/akram24/mall-customers)


## Project Directory Structure
```
Folder PATH listing
Volume serial number is A0F4-8B34
C:.
|   .flake8con
|   .gitignore
|   main.py
|   README.md
|   requirements.txt
|   test.py
|   testdev-ML.txt
|   
+---data
|       customers.csv
|       FILO2018_DEC_COM.xls
|       pop-sexe-age-quinquennal6817.xls
|       
+---outputs
|   |   population_perct_15_24_age.npy
|   |   
|   \---visuals
|           age_pyramid.png
|           bivariate_analysis.png
|           bivariate_analysis_subset.png
|           inhabitants_histogram.png
|           inhabitants_histogram_95p.png
|           optimal_value_of_k.png
|           
\---src
    |   __init__.py
    |   
    +---tasks
    |       _bivariate_analysis.py
    |       _cluster_analysis.py
    |       _univariate_analysis.py
    |       _visualization.py
    |       __init__.py
    |       
    \---utils
            __init__.py
```

## Conclusion
Observations about each task has been written as a comment inside the 
python file at the appropriate place and result has been stored inside 
`outputs` directory.



