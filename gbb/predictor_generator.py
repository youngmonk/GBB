import numpy
import pandas
from sklearn.linear_model import Ridge
import concurrent.futures


class GBBPredictor(object):

    def __init__(self):
        self.txn = pandas.read_csv('txn.csv')
        self.variant_mapper = pandas.read_csv('MappingPricer.csv')
        self.variant_mapper = self.variant_mapper[['Variant', 'Variant_Updated']]

    # private method
    def __preprocess_transactions__(self, txn):
        txn['Model'] = txn['Model'].str.upper()
        txn['Variant'] = txn['Variant'].str.upper()
        txn['City'] = txn['City'].str.upper()

        # mapping in variants
        mapping = self.variant_mapper.set_index('Variant').to_dict()
        # return mapping for x or x by default
        txn['Variant'] = txn['Variant'].apply(lambda x: mapping['Variant_Updated'].get(x, x))

        txn['key'] = txn['Model'] + "$" + txn['Variant'] + "$" + txn['City']
        txn['Age'] = txn['Transaction_Year'] - txn['Year']

        # removing unnecessary columns
        txn = txn[['key', 'Year', 'Ownership', 'Out_Kms', 'Age', 'Sold_Price']]

        return txn

    def __train_and_generate__(self, inputKey):
        training_data = self.txn[self.txn['key'] == inputKey]
        bucketedRes = pandas.DataFrame(
            columns=['Model', 'Variant', 'City', 'Ownership', 'Year', 'Out_Kms', 'key', 'Age', 'predPrice'])
        rowCnt = 0

        if training_data.size < 15:
            return bucketedRes

        features = training_data[['Year', 'Ownership', 'Out_Kms', 'Age']].as_matrix()
        labels = training_data['Sold_Price'].as_matrix()

        clf = Ridge()
        clf.fit(features, labels)

        for testYear in range(2004, 2016):
            for testOutKms in range(10000, 160000, 10000):
                testAge = 2016 - testYear
                inputSample = numpy.array([testYear, 1, testOutKms, testAge])
                inputSample = inputSample.reshape(1, -1)
                predictedPrice = clf.predict(inputSample)
                predictedPrice = round(predictedPrice[0])
                bucketedRes.set_value(rowCnt, 'Model', inputKey.split('$')[0])
                bucketedRes.set_value(rowCnt, 'Variant', inputKey.split('$')[1])
                bucketedRes.set_value(rowCnt, 'City', inputKey.split('$')[2])
                bucketedRes.set_value(rowCnt, 'Ownership', 1)
                bucketedRes.set_value(rowCnt, 'Year', testYear)
                bucketedRes.set_value(rowCnt, 'Out_Kms', testOutKms)
                bucketedRes.set_value(rowCnt, 'key', inputKey)
                bucketedRes.set_value(rowCnt, 'Age', testAge)
                bucketedRes.set_value(rowCnt, 'predPrice', predictedPrice)
                rowCnt += 1

        print('Finished for ' + inputKey + ". ")
        return bucketedRes

    def train_and_generate(self):
        self.txn = self.__preprocess_transactions__(self.txn)

        uniqueKeys = self.txn['key'].unique()

        result = pandas.DataFrame(
            columns=['Model', 'Variant', 'City', 'Ownership', 'Year', 'Out_Kms', 'key', 'Age', 'predPrice'])

        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            for bucket_output in executor.map(self.__train_and_generate__, uniqueKeys):
                result = pandas.concat([result, bucket_output], ignore_index=True)

        result.to_csv('public/result_python3.csv', sep=',')