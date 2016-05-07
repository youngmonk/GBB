import sys
from gbb import preprocess as pgr
from gbb import postprocess as ppr
import unittest
import pandas
import numpy as np
from gbb.mapper import Mapper
import logging


class TestPredictorGenerator(unittest.TestCase):

    def setUp(self):
        self.txn = pandas.DataFrame({'Make': ['Maruti', 'Maruti', 'Maruti'],
                                'Model': ['Alto', 'Alto', 'Alto 800'], 'Variant': ['LS BSIII', 'LS Sunroof', 'LS'],
                                'City': ['Bangalore', 'Bangalore', 'Bangalore'], 'Transaction_Year': [2010, 2010, 2010],
                                'Year': [2008, 2008, 2008], 'Ownership': [1, 1, 1], 'Out_Kms': [25500, 30000, 28000],
                                'Sold_Price': [200000, 200000, 200000]})

        self.variant_mapper = pandas.DataFrame({'Model': ['Alto', 'Alto', 'Alto 800'], 'Model_Updated': ['Alto', 'Alto', 'Alto'],
                                       'Variant': ['LS BSIII', 'LS Sunroof', 'LS'],
                                       'Variant_Updated': ['LS', 'LS', 'LS'],
                                       'Mappingin_factor': [1.2, 0.8, 1], 'Reversemapping_factor': [0.8, 1.2, 1]})

        self.mapper = pgr.transform_variant_mapper(self.variant_mapper)


    def test_preprocessing(self):
        log = logging.getLogger("TestPredictorGenerator.test_preprocessing")
        txn = pgr.preprocess_transactions(txn=self.txn, mapper=self.mapper)

        sold_price_transformed = txn['Sold_Price'].as_matrix()
        key_transformed = txn['key'].as_matrix()
        log.debug(txn)
        np.testing.assert_array_equal(sold_price_transformed, [240000.0,  160000.0, 200000.0])
        np.testing.assert_array_equal(key_transformed, ['ALTO$LS$BANGALORE',  'ALTO$LS$BANGALORE', 'ALTO$LS$BANGALORE'])


    def test_postprocessing(self):
        log = logging.getLogger("TestPredictorGenerator.test_preprocessing")
        result = pandas.DataFrame({'model': ['ALTO', 'ALTO', 'ALTO', 'ALTO'], 'version': ['LS', 'LS', 'LS', 'LS'],
                                   'city': ['BANGALORE', 'BANGALORE', 'BANGALORE', 'BANGALORE'],
                                   'ownership': [1, 1, 1, 1], 'year': [2008, 2009, 2010, 2011], 'age': [8, 7, 6, 5],
                                   'key': ['ALTO$LS$BANGALORE', 'ALTO$LS$BANGALORE', 'ALTO$LS$BANGALORE', 'ALTO$LS$BANGALORE'],
                                   'kms': [10000, 10000, 10000, 10000], 'make': ['Maruti', 'Maruti', 'Maruti', 'Maruti'],
                                   'good_price': [200000, 200000, 200000, 200000]})
        result = ppr.postprocess_predictions(result, self.mapper)
        log.debug(result)
        self.assertEqual(len(result.index), 16)
        self.assertEqual(len(result[result['model'] == 'ALTO'].index), 12)
        self.assertEqual(len(result[result['model'] == 'ALTO 800'].index), 4)
        self.assertEqual(len(result[result['version'] == 'LS'].index), 8)
        self.assertEqual(len(result[result['version'] == 'LS SUNROOF'].index), 4)
        self.assertEqual(len(result[result['version'] == 'LS BSIII'].index), 4)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestPredictorGenerator.test_preprocessing").setLevel(logging.DEBUG)
    unittest.main()
