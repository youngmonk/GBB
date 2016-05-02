import numpy
import pandas
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import concurrent.futures
from sklearn.preprocessing import StandardScaler
import time

LOG_FLAG = True
NORMALIZATION_FLAG = False
LINEAR = True


def generate_buckets():
    bucket_data = []

    for testYear in range(2004, 2016):
        for testOutKms in range(10000, 160000, 10000):
            testAge = 2016 - testYear
            inputSample = [testYear, 1, testOutKms, testAge]
            bucket_data.append(inputSample)

    return numpy.array(bucket_data)


def transform_variant_mapper(variant_mapper):
    variant_mapper = variant_mapper[['Variant', 'Variant_Updated', 'Mappingin_factor', 'Reversemapping_factor']]
    variant_mapper['Variant'] = variant_mapper['Variant'].str.upper()
    variant_mapper['Variant_Updated'] = variant_mapper['Variant_Updated'].str.upper()
    variant_price_mapping = variant_mapper.set_index('Variant').to_dict()

    variant_mapping = variant_price_mapping['Variant_Updated']
    price_mapping = variant_price_mapping['Mappingin_factor']
    reverse_price_mapping = variant_price_mapping['Reversemapping_factor']
    return variant_mapping, price_mapping, reverse_price_mapping


def preprocess_transactions(txn, price_mapping, variant_mapping):
    txn['Model'] = txn['Model'].str.upper()
    txn['Variant'] = txn['Variant'].str.upper()
    txn['City'] = txn['City'].str.upper()

    # scaling prices as per features in variants
    scaled_price = []
    for row in txn[['Variant', 'Sold_Price']].as_matrix():
        scaled_price.append(row[1]*price_mapping.get(row[0], 1))
    txn['Sold_Price'] = scaled_price

    # mapping in variants
    # if no mapping present return the variant as it is
    txn['Variant'] = txn['Variant'].apply(lambda x: variant_mapping.get(x, x))

    txn['key'] = txn['Model'] + "$" + txn['Variant'] + "$" + txn['City']
    txn['Age'] = txn['Transaction_Year'] - txn['Year']

    # removing unnecessary columns
    txn = txn[['key', 'Year', 'Ownership', 'Out_Kms', 'Age', 'Sold_Price']]

    return txn


class GBBPredictor(object):

    def __init__(self):
        self.txn = pandas.read_csv('txn.csv')

        # load mapping
        variant_mapper = pandas.read_csv('MappingPricer.csv')
        self.variant_mapping, self.price_mapping, self.reverse_price_mapping = transform_variant_mapper(variant_mapper)

        self.bucketed_queries = generate_buckets()

    def __postprocess_predictions__(self, result):
        inv_map = {}
        for k, v in self.variant_mapping.items():
            if k != v:
                inv_map[v] = inv_map.get(v, [])
                inv_map[v].append(k)

        # add new rows for inverse mapping
        for variant in inv_map:
            res_subset = result[result['Variant'] == variant].copy(deep=True)
            similar_variants = inv_map[variant]

            for similar_variant in similar_variants:
                similar_variant_data = res_subset.copy(deep=True)
                similar_variant_data['Variant'] = similar_variant
                similar_variant_data['predPrice'] *= self.reverse_price_mapping[similar_variant]
                result = pandas.concat([result, similar_variant_data], ignore_index=True)

        return result

    # Trains model for particular model, version and city. Generates data for
    # different bins of mileage and year of manufacturing
    # Returns a tuple of result and error
    def __train_and_generate__(self, inputKey):
        training_data = self.txn[self.txn['key'] == inputKey]
        bucketedRes = pandas.DataFrame(
            columns=['Model', 'Variant', 'City', 'Ownership', 'Year', 'Out_Kms', 'key', 'Age', 'predPrice'])
        errors = pandas.DataFrame(columns=['Key', 'Msg', 'Count'])

        try:
            if len(training_data.index) < 15:
                errors.set_value(0, 'Key', inputKey)
                errors.set_value(0, 'Msg', 'Less than 15 samples')
                errors.set_value(0, 'Count', len(training_data.index))
                return errors, None

            features = training_data[['Year', 'Ownership', 'Out_Kms', 'Age']].as_matrix()
            labels = training_data['Sold_Price'].as_matrix()
            bucketed_queries = self.bucketed_queries

            if LOG_FLAG:
                labels = numpy.log(labels)

            if NORMALIZATION_FLAG:
                feature_scaler = StandardScaler().fit(features)
                label_scaler = StandardScaler().fit(labels)
                features = feature_scaler.transform(features)
                labels = label_scaler.transform(labels)
                bucketed_queries = feature_scaler.transform(bucketed_queries)

            if LINEAR:
                clf = Ridge(fit_intercept=True, normalize=True).fit(features, labels)
            else:
                clf = SVR(C=100, gamma=0.001, epsilon=0.001, kernel='rbf').fit(features, labels)

            label_pred = clf.predict(bucketed_queries)

            # denormalize results
            if NORMALIZATION_FLAG:
                label_pred = label_pred*label_scaler.scale_ + label_scaler.mean_

            if LOG_FLAG:
                label_pred = numpy.exp(label_pred)
                label_pred = numpy.round(label_pred)

            bucketedRes['Year'] = self.bucketed_queries[:, 0]
            bucketedRes['Ownership'] = self.bucketed_queries[:, 1]
            bucketedRes['Out_Kms'] = self.bucketed_queries[:, 2]
            bucketedRes['Age'] = self.bucketed_queries[:, 3]
            bucketedRes['Model'] = inputKey.split('$')[0]
            bucketedRes['Variant'] = inputKey.split('$')[1]
            bucketedRes['City'] = inputKey.split('$')[2]
            bucketedRes['predPrice'] = label_pred

            print('Finished for ' + inputKey + ". ")
            return None, bucketedRes
        except:
            print('Exception for ' + inputKey + ". ")
            errors.set_value(0, 'Key', inputKey)
            errors.set_value(0, 'Msg', ' error in data')
            return errors, None

    def train_and_generate(self):
        self.txn = preprocess_transactions(self.txn, self.price_mapping, self.variant_mapping)

        uniqueKeys = self.txn['key'].unique()

        result = pandas.DataFrame(
            columns=['Model', 'Variant', 'City', 'Ownership', 'Year', 'Out_Kms', 'key', 'Age', 'predPrice'])
        errors = pandas.DataFrame(columns=['Key', 'Msg', 'Count'])

        # with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        #     for (err, bucket_output) in executor.map(self.__train_and_generate__, uniqueKeys):
        #         if bucket_output is not None:
        #             result = pandas.concat([result, bucket_output], ignore_index=True)
        #         if err is not None:
        #             errors = pandas.concat([errors, err], ignore_index=True)

        for (err, bucket_output) in map(self.__train_and_generate__, uniqueKeys):
            if bucket_output is not None:
                result = pandas.concat([result, bucket_output], ignore_index=True)
            if err is not None:
                errors = pandas.concat([errors, err], ignore_index=True)

        result = self.__postprocess_predictions__(result)
        result.to_csv('public/result_python3.csv', sep=',')
        errors.to_csv('public/training_errors.csv', sep=',')


# start_time = time.time()
# GBBPredictor().train_and_generate()
# finish_time = time.time()
# print('Total time : ', finish_time-start_time, 'secs')