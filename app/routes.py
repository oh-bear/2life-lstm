# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from app import app
from flask import request, jsonify
#from Sentiment_svm import svm_predict
try:
    from Sentiment_lstm import lstm_predict
except Exception as e:
    app.logger.error(e)

@app.route('/ner', methods=['POST'])
def ner():
    app.logger.info('request for /ner')
    content = request.form.get('content')
    if content:
        try:
            data = lstm_predict(content)
            #data = content
            return jsonify(code=0, message='success', data=data)
        except Exception as e:
            app.logger.error(e)
            return jsonify(code=0, message='predict failed', data={})
    else:
        return jsonify(code=-1, message='content required', data={})
