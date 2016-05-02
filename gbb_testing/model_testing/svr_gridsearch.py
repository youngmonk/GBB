import numpy
import pandas
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import concurrent.futures
import time
from sklearn.cross_validation import KFold
import math
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


def read_txn(transaction_file='txn.csv', mapping_file='MappingPricer.csv'):
    txn = pandas.read_csv('txn.csv')
    variant_mapper = pandas.read_csv('MappingPricer.csv')
    variant_mapper = variant_mapper[['Variant', 'Variant_Updated']]
    # standardize variants
    variant_mapper['Variant'] = variant_mapper['Variant'].str.upper()
    variant_mapper['Variant_Updated'] = variant_mapper['Variant_Updated'].str.upper()
    # convert variants to dictionary
    mapping = variant_mapper.set_index('Variant').to_dict()
    mapping = mapping['Variant_Updated']
    return txn, mapping


def preprocess_txn(txn, mapping, log_flag=True):
    txn['Model'] = txn['Model'].str.upper()
    txn['Variant'] = txn['Variant'].str.upper()
    txn['City'] = txn['City'].str.upper()

    # mapping in variants
    # if no mapping present return the variant as it is
    txn['Variant'] = txn['Variant'].apply(lambda x: mapping.get(x, x))

    txn['key'] = txn['Model'] + "$" + txn['Variant'] + "$" + txn['City']
    txn['Age'] = txn['Transaction_Year'] - txn['Year']

    # removing unnecessary columns and handle categorical data
    features = txn[['key', 'Year', 'Ownership', 'Out_Kms', 'Age']]

    # taking logarithm of price
    labels = txn[['Sold_Price']]
    if log_flag:
        labels = labels.apply(numpy.log10)

    return features, labels


def get_awesome_params():
    txn, mapping = read_txn()
    features, labels = preprocess_txn(txn, mapping, log_flag=LOG_FLAG)

    swift_features = features[features['key'] == 'FIGO$DURATORQ DIESEL TITANIUM 1.4$BANGALORE']
    swift_labels = labels[features['key'] == 'FIGO$DURATORQ DIESEL TITANIUM 1.4$BANGALORE']

    swift_features = swift_features[['Year', 'Ownership', 'Out_Kms', 'Age']].as_matrix()
    swift_labels = swift_labels.as_matrix()

    # normalizing prices
    # feature_scaler = StandardScaler().fit(swift_features)
    # labels_scaler = StandardScaler().fit(swift_labels)
    # swift_features = feature_scaler.transform(swift_features)
    # swift_labels = labels_scaler.transform(swift_labels).ravel()

    C_range = 10.0 ** numpy.arange(-4, 4)
    gamma_range = 10.0 ** numpy.arange(-4, 4)
    epsilon_range = 10.0 ** numpy.arange(-4, 4)
    param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist(), epsilon=epsilon_range)
    svr = SVR()
    grid = GridSearchCV(svr, param_grid)
    grid.fit(swift_features, swift_labels)
    print("The best classifier is: ", grid.best_estimator_)
    print(grid.grid_scores_)

start_time = time.time()
LOG_FLAG=True
get_awesome_params()
finish_time = time.time()

print("Finished in " + str(finish_time - start_time) + " secs")
