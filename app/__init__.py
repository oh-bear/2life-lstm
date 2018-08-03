# coding: utf-8
from __future__ import unicode_literals
import logging
from logging.handlers import SMTPHandler, RotatingFileHandler
from flask import Flask
import os

app = Flask(__name__)

# set file logs
if not app.debug and not app.testing:
    if not os.path.exists('logs'):
        os.mkdir('logs')

    file_handler = RotatingFileHandler('logs/twoLife.log',
                                       maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('twoLife ner server startup')

from app import routes
