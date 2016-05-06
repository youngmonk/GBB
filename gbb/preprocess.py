import numpy
import pandas
import time


def transform_variant_mapper(variant_mapper):
    variant_mapper = variant_mapper[['Model', 'Model_Updated', 'Variant', 'Variant_Updated', 'Mappingin_factor',
                                     'Reversemapping_factor']]
    variant_mapper['Model'] = variant_mapper['Model'].str.upper()
    variant_mapper['Model_Updated'] = variant_mapper['Model_Updated'].str.upper()
    model_mapping = variant_mapper.set_index('Model').to_dict()['Model_Updated']

    variant_mapper['Variant'] = variant_mapper['Variant'].str.upper()
    variant_mapper['Variant_Updated'] = variant_mapper['Variant_Updated'].str.upper()
    variant_price_mapping = variant_mapper.set_index('Variant').to_dict()

    variant_mapping = variant_price_mapping['Variant_Updated']
    price_mapping = variant_price_mapping['Mappingin_factor']
    reverse_price_mapping = variant_price_mapping['Reversemapping_factor']
    return variant_mapping, price_mapping, reverse_price_mapping, model_mapping


def preprocess_models(txn, model_mapping):
    txn['Model'] = txn['Model'].apply(lambda x: model_mapping.get(x, x))
    return txn


def preprocess_variants(txn, price_mapping, variant_mapping):
    # scaling prices as per features in variants
    scaled_price = []
    for row in txn[['Variant', 'Sold_Price']].as_matrix():
        scaled_price.append(row[1] * price_mapping.get(row[0], 1))
    txn['Sold_Price'] = scaled_price

    # mapping in variants
    # if no mapping present return the variant as it is
    txn['Variant'] = txn['Variant'].apply(lambda x: variant_mapping.get(x, x))
    return txn


def preprocess_transactions(txn, price_mapping, variant_mapping, model_mapping):
    txn['Make'] = txn['Make'].str.upper()
    txn['Model'] = txn['Model'].str.upper()
    txn['Variant'] = txn['Variant'].str.upper()
    txn['City'] = txn['City'].str.upper()

    txn = preprocess_models(txn, model_mapping)
    txn = preprocess_variants(txn, price_mapping, variant_mapping)

    txn['key'] = txn['Model'] + "$" + txn['Variant'] + "$" + txn['City']
    txn['Age'] = txn['Transaction_Year'] - txn['Year']

    # removing unnecessary columns
    txn = txn[['Make', 'key', 'Year', 'Ownership', 'Out_Kms', 'Age', 'Sold_Price']]

    return txn
