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

        train_request = request.get_json(force=True)

        learner_type = train_request['learner']
        predictor_generator.LINEAR = learner_type == 'ridge'
        predictor_generator.NORMALIZATION_FLAG = learner_type == 'svr'

        training_file = train_request['trainingFile']
        mapping_file = train_request['mappingFile']

        start_time = time.time()
        predictor_generator.GBBPredictor(training_file=training_file, mapping_file=mapping_file).train_and_generate()
        finish_time = time.time()

        app.logger.info("Finished in " + str(finish_time - start_time) + " secs")
        return 'success'

    @app.route('/save_predicted_prices', methods=["POST"])
    def save_predicted_prices():
        app.logger.info("Saving predicted prices")
        PREDICTOR_TASKS.append({'task_date': int(time.time()), 'task': 'Saving predicted prices'})

        save_request = request.get_json(force=True)
        vehicle_type = save_request['vehicle_type']

        print('Updating db for ', vehicle_type)
        pred_res = pd.read_csv('public/result_python3.csv')
        # compute md5 hashes
        pred_res['md5'] = pred_res['make'] + pred_res['model'] + pred_res['version'] + pred_res['city'] + \
                          pred_res['year'].apply(str) + pred_res['kms'].apply(str)
        pred_res['md5'] = [hashlib.md5(val.encode('utf-8')).hexdigest() for val in pred_res['md5']]
        pred_res['vehicle_type'] = vehicle_type

        predicted_prices_dicts = pred_res.to_dict('records')
        
        start_time = time.time()
        PredictedPrices.save_bulk(predicted_prices_dicts, vehicle_type)
        finish_time = time.time()

        app.logger.info("Finished insertion in " + str(finish_time - start_time) + " secs")
        return 'success'
