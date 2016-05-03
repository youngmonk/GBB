from flask import Flask, send_from_directory
from routes import predictor
from gevent.wsgi import WSGIServer
import logging

app = Flask(__name__, static_url_path='/public')
UPLOAD_FOLDER = ''

# app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return send_from_directory('public', 'index.html')


@app.route('/loaderio-41a415f519dced3ce153f1a9fe518a17/', methods=['GET'])
def verify_loaderio():
    return 'loaderio-41a415f519dced3ce153f1a9fe518a17'


@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('public', path)

# initialize all routes
predictor.init(app=app)

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=7000) # debug server
    print('Starting server')
    http_server = WSGIServer(('', 7000), app)
    http_server.serve_forever()
