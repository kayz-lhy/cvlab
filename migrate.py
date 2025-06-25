#!/usr/bin/env python
import os
from flask.cli import FlaskGroup
from app import create_app

app = create_app(os.environ.get('FLASK_ENV', 'default'))
cli = FlaskGroup(app)

if __name__ == '__main__':
    cli()