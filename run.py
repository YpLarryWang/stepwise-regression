# this script is a example of using stepwise regression to select features
# and then use the selected features to train a model
import logging
import pandas as pd
import numpy as np
from stepwise_regression import StepwiseRegressionSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

data_dir = './data/'
log_dir = f'./stepwise_example.log'

logging.basicConfig(filename=log_dir,
                    level=logging.INFO,
                    filemode='w',  # default is 'a' change to 'w' to overwrite
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load data
train_data = pd.read_csv(data_dir + 'train.csv')

# split data into train and valid
train, valid = train_test_split(train_data, test_size=0.2, random_state=42)

# for simplicity, we only use numeric predictors
train_X = train.select_dtypes(exclude=['object'])
valid_X = valid.select_dtypes(exclude=['object'])

# handle inf, -inf, replace with nan
train_X = train_X.replace([np.inf, -np.inf], np.nan)

# we can either drop the rows with NaN or fill the NaN with 0 or mean
# here we fill the NaN with mean
train_X = train_X.fillna(train_X.mean())
valid_X = valid_X.fillna(valid_X.mean())

# intantiate a stepwise regression selector
selector = StepwiseRegressionSelector(
    data=train_X, target='SalePrice', initial_list=None, threshold_in=0.05, threshold_out=0.10, verbose=True)

# select features
selected_features = selector.forward_backward()

# train a linear regression model
lr = LinearRegression()
lr.fit(train_X[selected_features], train_X['SalePrice'])

# predict on test set
pred = lr.predict(valid_X[selected_features])

# compute r^2 and rmse
logger.info('r^2: {:.4}'.format(r2_score(valid_X['SalePrice'], pred)))
logger.info('rmse: {:.4}'.format(
    np.sqrt(mean_squared_error(valid_X['SalePrice'], pred))))
