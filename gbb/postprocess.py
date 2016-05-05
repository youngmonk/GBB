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
    for variant in inv_map:
        res_subset = result[result['version'] == variant].copy(deep=True)
        similar_variants = inv_map[variant]

        for similar_variant in similar_variants:
            similar_variant_data = res_subset.copy(deep=True)
            similar_variant_data['version'] = similar_variant
            similar_variant_data['good_price'] *= reverse_price_mapping[similar_variant]
            result = pandas.concat([result, similar_variant_data], ignore_index=True)

    return result


def postprocess_models(result, model_mapping):
    inv_map = inverse_map(model_mapping)

    # add new rows for inverse mapping
    for model in inv_map:
        res_subset = result[result['model'] == model].copy(deep=True)
        similar_variants = inv_map[model]

        for similar_variant in similar_variants:
            similar_variant_data = res_subset.copy(deep=True)
            similar_variant_data['model'] = similar_variant
            result = pandas.concat([result, similar_variant_data], ignore_index=True)

    return result


def postprocess_predictions(result, variant_mapping, reverse_price_mapping, model_mapping):
    print('Postprocessing transactions')
    result = postprocess_models(result, model_mapping)
    result = postprocess_variants(result, variant_mapping, reverse_price_mapping)
    print('Postprocessing finished')
    return result
