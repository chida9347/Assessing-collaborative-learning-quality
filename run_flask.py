#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask_app.app import app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)
