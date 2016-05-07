import pandas


def inverse_map(key_map):
    inv_map = {}
    for k, v in key_map.items():
        if k != v:
            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)

    return inv_map


def postprocess_variants(result, variant_mapping, reverse_price_mapping):
    inv_map = inverse_map(variant_mapping)

    # add new rows for inverse mapping
    for model_variant in inv_map:
        res_subset = result[result['model_version_updated'] == model_variant].copy(deep=True)
        similar_model_variants = inv_map[model_variant]

        for similar_model_variant in similar_model_variants:
            similar_variant_data = res_subset.copy(deep=True)

            similar_model, similar_version = similar_model_variant.split('$')
            similar_variant_data['version'] = similar_version
            similar_variant_data['model'] = similar_model
            similar_variant_data['good_price'] *= reverse_price_mapping[similar_model_variant]
            result = pandas.concat([result, similar_variant_data], ignore_index=True)

    return result


def postprocess_predictions(result, mapper):
    print('Postprocessing transactions')
    result['model_version_updated'] = result['model'] + '$' + result['version']
    result = postprocess_variants(result, mapper.variant_mapping, mapper.reverse_price_mapping)
    print('Postprocessing finished')
    return result
