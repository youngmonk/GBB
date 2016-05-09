class Config(object):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:postgres@localhost:5432/gozoomo?application_name=GBB'
    SQLALCHEMY_POOL_SIZE = 10
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    POOL_SIZE = 10
    UPLOAD_FOLDER = ''


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:gaeK1oo6ai@pg.gozoomo.com:5432/gozoomo?application_name=GBB'
