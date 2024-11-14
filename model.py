from dataset_preprocessing import *
from constants import SALE_PRICE_KEY, ID_KEY
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

# format files
train_dataset = pd.read_csv('data/train.csv')
test_dataset = pd.read_csv('data/test.csv')
format_dataset(train_dataset, 'train.csv')
format_dataset(test_dataset, 'test.csv')

# one-hot encoding
combined_data = pd.concat([train_dataset, test_dataset], keys=['train', 'test'])
combined_dummies = pd.get_dummies(combined_data, columns=categorical_data_keys)
train_dataset = combined_dummies.xs('train')
test_dataset = combined_dummies.xs('test')

x = train_dataset.drop(columns=[SALE_PRICE_KEY])
y = train_dataset[SALE_PRICE_KEY]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# choose the best model
model = CatBoostRegressor(verbose=False)
model = model.fit(x_train, y_train.values.ravel())

y_pred = model.predict(x_test)
mse = np.mean(np.sqrt(-cross_val_score(model, x_test, y_test.to_numpy(), cv=5, scoring="neg_mean_squared_error")))
print(f'MSE CatBoostRegressor: {mse}')

# graph for comparing
plt.figure()
plt.scatter(range(len(y_test)), y_test, s=7, label="y_test")
plt.scatter(range(len(y_test)), y_pred, s=7, label="y_pred")
plt.legend()
plt.show()

# creating file with result predicted data
test_prediction = model.predict(test_dataset)

result_dataset = pd.DataFrame({ID_KEY: range(len(test_dataset)),
                               SALE_PRICE_KEY: test_prediction})
create_result_file(result_dataset)
