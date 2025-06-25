from app import create_app
import os

app = create_app(os.environ.get('FLASK_ENV', 'default'))

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)