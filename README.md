# ML Apartment Price Prediction

Model for sale price predicting.

## Libraries

- catboost
- matplotlib
- numpy
- pandas
- sklearn
- xgboost

## Data for prediction
### Data

Data description: ```data_description.txt```.

Train data: ```data/train.csv```.

Test data: ```data/test.csv```.

Test example: ```data/sample_submission.csv```.

Test results: ```result/result.csv```.

### Formatted data

Formatted train and test datasets are in directory 
```formatted_data```

### Result

Predicted result sale price: ```result.csv```

## Program files

### main.py

- preprocessing data with one-hot encoding;
- choosing the best model (```CatBoostRegressor```) by comparing them;
- predicting data and comparing them;
- showing graph for comparing.
- creating result file;

### constants.py

Constants for keys & maps for data formatting. 
It's description can be found in ```data_description.txt```.

### dataset_preprocessing.py

```def format_dataset(df, filename)```
- processing na fields;
- deleting correlated fields

```def create_formatted_file(df, filename)```
- create formatted file in **formatted_data** directory.

```def create_result_file(df)```
- create file with results in **result** directory.

### correlation_processing.py

```def check_corr_emptiness(corr_coefficients)```
- check correlation coefficients for emptiness and in case of emptiness add if for deleting.

```def check_corr_coefficients(corr_coefficients)```
- find dependent values and add them for deleting.

### model_comparing.py

```def compare_models(x_train, y_train)```
- compare models for choosing the best option by mean squared error comparing.

```def print_feature_importances(column_keys, model)```
- print feature importances for current model.
