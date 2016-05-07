from flask import request, Response, send_from_directory
import os
from werkzeug import secure_filename
from routes.predictor import PREDICTOR_TASKS
import time
import json
from os import listdir
from os.path import isfile, join

ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = ''


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

    @app.route('/list_files', methods=['GET'])
    def get_files():
        app.logger.info("Getting file list")
        file_path = './'
        csv_files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and allowed_file(f)]
        return Response(json.dumps(csv_files), mimetype='application/json')

    @app.route('/delete_file/<file_name>', methods=['DELETE'])
    def delete_file(file_name):
        os.remove(file_name)
        return 'success'
