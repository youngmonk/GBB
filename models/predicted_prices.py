from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictedPrices(db.Model):
    __tablename__ = 'predicted_prices'

    id = db.Column(db.Integer, primary_key=True)
    make = db.Column(db.String(80))
    model = db.Column(db.String(80))
    version = db.Column(db.String(80))
    city = db.Column(db.String(80))
    ownership = db.Column(db.Integer)
    year = db.Column(db.Integer)
    kms = db.Column(db.Integer)
    age = db.Column(db.Integer)
    key = db.Column(db.String(80))
    good_price = db.Column(db.Integer)
    md5 = db.Column(db.String(80))

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def save(self):
        self.md5 = 'sdfdsf'
        db.session.add(self)
        db.session.commit()
