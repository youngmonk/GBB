from flask import request, Response, send_from_directory
import os
from werkzeug import secure_filename
import time
import json
from gbb import predictor_generator
from models.predicted_prices import PredictedPrices
import pandas as pd
import hashlib

PREDICTOR_TASKS = []


def init(app):

    @app.route('/get_tasks')
    def get_tasks():
        return Response(json.dumps(PREDICTOR_TASKS), mimetype='application/json')

    @app.route('/train_and_generate', methods=['POST'])
    def train_and_generate():
        app.logger.info("Training new model")
        PREDICTOR_TASKS.append({'task_date': int(time.time()), 'task': 'Training new model'})

        learner_type = request.get_json(force=True)['learner']
        predictor_generator.LINEAR = learner_type == 'ridge'
        predictor_generator.NORMALIZATION_FLAG = learner_type == 'svr'

        start_time = time.time()
        predictor_generator.GBBPredictor().train_and_generate()
        finish_time = time.time()

        app.logger.info("Finished in " + str(finish_time - start_time) + " secs")
        return 'success'

    @app.route('/save_predicted_prices', methods=["POST"])
    def save_predicted_prices():
        app.logger.info("Saving predicted prices")
        PREDICTOR_TASKS.append({'task_date': int(time.time()), 'task': 'Saving predicted prices'})

        pred_res = pd.read_csv('public/result_python3.csv')
        # compute md5 hashes
        pred_res['md5'] = pred_res['make'] + pred_res['model'] + pred_res['version'] + pred_res['city'] + \
                          pred_res['year'].apply(str) + pred_res['kms'].apply(str)
        pred_res['md5'] = [hashlib.md5(val.encode('utf-8')).hexdigest() for val in pred_res['md5']]

        predicted_prices_dicts = pred_res.to_dict('records')
        
        start_time = time.time()
        PredictedPrices.save_bulk(predicted_prices_dicts)
        finish_time = time.time()

        app.logger.info("Finished insertion in " + str(finish_time - start_time) + " secs")
        return 'success'
