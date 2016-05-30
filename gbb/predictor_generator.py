import numpy
import pandas
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import concurrent.futures
from sklearn.preprocessing import StandardScaler
import time
from gbb import preprocess as pgr
from gbb import postprocess as postpr

LOG_FLAG = True
NORMALIZATION_FLAG = False
LINEAR = True


def generate_buckets():
    bucket_data = []

    for testYear in range(2004, 2017):
        for testOutKms in range(10000, 160000, 10000):
            testAge = 2016 - testYear
            inputSample = [testYear, 1, testOutKms, testAge]
            bucket_data.append(inputSample)

    return numpy.array(bucket_data)


class GBBPredictor(object):

    def __init__(self, training_file='txn.csv', mapping_file='MappingPricer.csv'):
        self.txn = pandas.read_csv(training_file)

        # load mapping
        variant_mapper = pandas.read_csv(mapping_file)
        self.mapper = pgr.transform_variant_mapper(variant_mapper)

        self.bucketed_queries = generate_buckets()


    # Trains model for particular model, version and city. Generates data for
    # different bins of mileage and year of manufacturing
    # Returns a tuple of result and error
    def __train_and_generate__(self, inputKey):
        training_data = self.txn[self.txn['key'] == inputKey]
        bucketedRes = pandas.DataFrame(
            columns=['model', 'version', 'city', 'ownership', 'year', 'kms', 'key', 'age', 'good_price'])
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

            # keys correspond with database columns
            bucketedRes['year'] = self.bucketed_queries[:, 0]
            bucketedRes['ownership'] = self.bucketed_queries[:, 1]
            bucketedRes['kms'] = self.bucketed_queries[:, 2]
            bucketedRes['age'] = self.bucketed_queries[:, 3]
            bucketedRes['model'] = inputKey.split('$')[0]
            bucketedRes['version'] = inputKey.split('$')[1]
            bucketedRes['city'] = inputKey.split('$')[2]
            bucketedRes['key'] = inputKey
            bucketedRes['make'] = training_data['Make'].as_matrix()[0]
            bucketedRes['good_price'] = label_pred

            print('Finished for ' + inputKey + ". ")
            return None, bucketedRes
        except:
            print('Exception for ' + inputKey + ". ")
            errors.set_value(0, 'Key', inputKey)
            errors.set_value(0, 'Msg', ' error in data')
            return errors, None

    def train_and_generate(self):
        self.txn = pgr.preprocess_transactions(self.txn, self.mapper)

        uniqueKeys = self.txn['key'].unique()

        result = pandas.DataFrame()
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

        start_time = time.time()
        result = postpr.postprocess_predictions(result, self.mapper)
        end_time = time.time()
        print('Postprocessing time : ', end_time-start_time, ' secs')
        result.to_csv('public/result_python3.csv', sep=',')
        errors.to_csv('public/training_errors.csv', sep=',')


if __name__ == '__main__':
    start_time = time.time()
    GBBPredictor().train_and_generate()
    finish_time = time.time()
    print('Total time : ', finish_time-start_time, 'secs')
