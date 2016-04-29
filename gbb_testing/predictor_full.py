import numpy
import pandas
from sklearn.linear_model import Ridge
import concurrent.futures
import time
from sklearn.cross_validation import KFold
import math


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

    txn['key'] = txn['Model'] + "$" + txn['Variant'] + "$" + txn['Year'].apply(str)
    txn['Age'] = txn['Transaction_Year'] - txn['Year']

    # removing unnecessary columns and handle categorical data
    features = txn[['key', 'City', 'Ownership', 'Out_Kms', 'Age']]
    features = pandas.get_dummies(features)

    # taking logarithm of price
    labels = txn[['Sold_Price']]
    if log_flag:
        labels = labels.apply(numpy.log10)

    return features, labels


def compute_rmse(features, labels, train_index, test_index, log_flag=True):
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    clf = Ridge(fit_intercept=True, normalize=True).fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    if log_flag:
        actual_pred = numpy.array([10 ** y for y in y_pred])
        actual_price = numpy.array([10 ** y for y in y_test])
    else:
        actual_pred = y_pred
        actual_price = y_test

    actual_rmse_pc = numpy.sqrt(numpy.mean(((actual_pred - actual_price) / actual_price) ** 2))
    actual_rmse = numpy.sqrt(numpy.mean((actual_pred - actual_price) ** 2))

    return actual_rmse, actual_rmse_pc


def run_kfold():
    txn, mapping = read_txn()
    features, labels = preprocess_txn(txn, mapping, log_flag=LOG_FLAG)

    features_matrix = features.as_matrix()
    labels_matrix = labels.as_matrix()

    r, c = features_matrix.shape
    kf = KFold(n=r, n_folds=10, shuffle=False, random_state=None)

    k_rmse = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #     for train_index, test_index in kf:
    #         rmse = executor.submit(compute_rmse, features_matrix, labels_matrix, train_index, test_index)
    #         k_rmse.append(rmse)

    for train_index, test_index in kf:
        rmse = compute_rmse(features_matrix, labels_matrix, train_index, test_index, log_flag=LOG_FLAG)
        k_rmse.append(rmse)

    mean_rmse = 0
    for rmse, rmse_pc in k_rmse:
        print(rmse, rmse_pc)
        mean_rmse += rmse_pc

    mean_rmse = mean_rmse/len(k_rmse)
    print('Mean rmse : ', mean_rmse)

start_time = time.time()
LOG_FLAG=False
run_kfold()
finish_time = time.time()

print("Finished in " + str(finish_time - start_time) + " secs")
