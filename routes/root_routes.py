from flask import send_from_directory


def init(app):
    @app.route('/')
    def root():
        return send_from_directory('public', 'index.html')

    @app.route('/loaderio-41a415f519dced3ce153f1a9fe518a17/', methods=['GET'])
    def verify_loaderio():
        return 'loaderio-41a415f519dced3ce153f1a9fe518a17'

    @app.route('/assets/<path:path>')
    def send_assets(path):
        return send_from_directory('public', path)