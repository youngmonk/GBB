import numpy
import pandas
import time
from gbb.mapper import Mapper


def transform_variant_mapper(variant_mapper):
    variant_mapper = variant_mapper[['Model', 'Model_Updated', 'Variant', 'Variant_Updated', 'Mappingin_factor',
                                     'Reversemapping_factor', 'Pricevariation_range']]
    mapper = Mapper()
    variant_mapper['Model'] = variant_mapper['Model'].str.upper()
    variant_mapper['Model_Updated'] = variant_mapper['Model_Updated'].str.upper()
    variant_mapper['Variant'] = variant_mapper['Variant'].str.upper()
    variant_mapper['Variant_Updated'] = variant_mapper['Variant_Updated'].str.upper()

    variant_mapper['Model_Variant'] = variant_mapper['Model'] + '$' + variant_mapper['Variant']
    variant_mapper['Model_Variant_Updated'] = variant_mapper['Model_Updated'] + '$' + variant_mapper['Variant_Updated']

    variant_price_mapping = variant_mapper.set_index('Model_Variant').to_dict()

    mapper.variant_mapping = variant_price_mapping['Model_Variant_Updated']
    mapper.price_mapping = variant_price_mapping['Mappingin_factor']
    mapper.reverse_price_mapping = variant_price_mapping['Reversemapping_factor']
    mapper.price_variation_mapping = variant_price_mapping['Pricevariation_range']
    return mapper


# Scale pricing and transform model_variants to nearest mapping
def scale_pricing(txn, price_mapping, variant_mapping):
    # scaling prices as per features in variants
    scaled_price = []
    for row in txn[['tmp_key', 'Sold_Price']].as_matrix():
        scaled_price.append(row[1] * price_mapping.get(row[0], 1))
    txn['Sold_Price'] = scaled_price

    # mapping in model$variants
    # if no mapping present return the model$variant as it is
    txn['tmp_key'] = txn['tmp_key'].apply(lambda x: variant_mapping.get(x, x))
    return txn


def preprocess_transactions(txn, mapper):
    txn['Make'] = txn['Make'].str.upper()
    txn['Model'] = txn['Model'].str.upper()
    txn['Variant'] = txn['Variant'].str.upper()
    txn['City'] = txn['City'].str.upper()

    txn['tmp_key'] = txn['Model'] + '$' + txn['Variant']
    txn = scale_pricing(txn, mapper.price_mapping, mapper.variant_mapping)

    txn['key'] = txn['tmp_key'] + "$" + txn['City']
    txn['Age'] = txn['Transaction_Year'] - txn['Year']

    # removing unnecessary columns
    txn = txn[['Make', 'key', 'Year', 'Ownership', 'Out_Kms', 'Age', 'Sold_Price']]

    return txn
