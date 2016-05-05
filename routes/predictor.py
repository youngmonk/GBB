from flask import request, Response, send_from_directory
import os
from werkzeug import secure_filename
import time
import json
from gbb import predictor_generator
from models.predicted_prices import PredictedPrices
import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'jpg'}
UPLOAD_FOLDER = ''
PREDICTOR_TASKS = []


def init(app):

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

    @app.route('/upload_training_data', methods=['POST'])
    def upload_training_data():
        app.logger.info("Uploading new training data")
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            PREDICTOR_TASKS.append({'task_date': int(time.time()), 'task': 'Uploaded ' + filename})
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'success'

        return Response(status=403)

    @app.route('/test')
    def test():
        PREDICTOR_TASKS.append({'task_date': int(time.time()), 'task': 'Test task'})
        return 'success'

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

        predicted_price_buckets = pd.read_csv('public/result_python3.csv')
        predicted_prices_dicts = predicted_price_buckets.to_dict('records')
        
        start_time = time.time()
        PredictedPrices.save_bulk(predicted_prices_dicts)
        finish_time = time.time()

        app.logger.info("Finished insertion in " + str(finish_time - start_time) + " secs")
        return 'success'
