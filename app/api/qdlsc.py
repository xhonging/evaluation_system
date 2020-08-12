from flask import Blueprint
from flask import jsonify


qdlsc_bp = Blueprint('qdlsc', __name__)


@qdlsc_bp.route('/station/run')
def run():
    pass
    # df = test_1()
    # return jsonify(df)

