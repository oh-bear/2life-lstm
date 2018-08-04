# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from app import app
from flask import request, jsonify
from config import Config
import json

try:
    from Sentiment_lstm import lstm_predict
except Exception as e:
    app.logger.error(e)

@app.route('/ner', methods=['POST'])
def ner():
    app.logger.info('request for /ner')
    data = None
    # content = request.json.get('content')
    request_data = json.loads(request.data) 
    content = request_data.get('content')
    key = request_data.get('key')
    if key != Config.SECRET_KEY:
        code, message = 401, 'Unauthorized'
    elif content:
        try:
            data = lstm_predict(content)
            #data = content
            code, message = 0, 'success' 
        except Exception as e:
            app.logger.error(e)
            code, message = 503, 'LSTM failed'
    else:
        code, message = 400, '"content" required'

    return jsonify(code=code, message=message, data=data)
