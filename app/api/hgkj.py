from flask import Blueprint
from flask import jsonify
# from ..hg.test import test_1


hgkj_bp = Blueprint('hgkj', __name__)


@hgkj_bp.route('/station/run')
def run():
    pass
    # df = test_1()
    # return jsonify(df)

