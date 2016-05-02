# import sys
# sys.path.append('/home/sainageswar/youngmonk/GBBpy')
from gbb import predictor_generator as pgr
import unittest
import pandas
import numpy as np


class TestPredictorGenerator(unittest.TestCase):
    def test_preprocessing(self):
        txn = pandas.DataFrame({'Model': ['Alto', 'Alto'], 'Variant': ['LS BSIII', 'LS Sunroof'],
                                'City': ['Bangalore', 'Bangalore'], 'Transaction_Year': [2010, 2010],
                                'Year': [2008, 2008], 'Ownership': [1, 1], 'Out_Kms': [25500, 30000],
                                'Sold_Price': [200000, 200000]})
        variant_mapper = pandas.DataFrame({'Variant': ['LS BSIII', 'LS Sunroof'], 'Variant_Updated': ['LS', 'LS'],
                                           'Mappingin_factor': [1.2, 0.8], 'Reversemapping_factor': [0.8, 1.2]})

        variant_mapping, price_mapping, reverse_price_mapping = pgr.transform_variant_mapper(variant_mapper)
        txn = pgr.preprocess_transactions(txn=txn, variant_mapping=variant_mapping, price_mapping=price_mapping)

        sold_price_transformed = txn['Sold_Price'].as_matrix()
        key_transformed = txn['key'].as_matrix()
        print(txn)
        np.testing.assert_array_equal(sold_price_transformed, [240000.0,  160000.0])
        np.testing.assert_array_equal(key_transformed, ['ALTO$LS$BANGALORE',  'ALTO$LS$BANGALORE'])


if __name__ == '__main__':
    unittest.main()
