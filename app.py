from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
from iris_auth import IrisAuthenticationSystem
import logging
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)

eye_model_path = 'weights/best (8).pt'
iris_model_path = 'weights/best (7).pt'

auth_system = IrisAuthenticationSystem(eye_model_path, iris_model_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 200

    app.logger.debug("Registration request received")
    data = request.json
    user_id = data['userId']
    image_data = data['image'].split(',')[1]
    app.logger.debug(f"User ID: {user_id}")
    app.logger.debug(f"Image data length: {len(image_data)}")

    try:
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        app.logger.debug(f"Decoded image shape: {image.shape}")

        success, message = auth_system.register_user(user_id, image)
        app.logger.debug(f"Registration result: success={success}, message={message}")
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        app.logger.error(f"Error during registration: {str(e)}")
        return jsonify({'success': False, 'message': f"Error during registration: {str(e)}"})


@app.route('/authenticate', methods=['POST', 'OPTIONS'])
def authenticate():
    if request.method == 'OPTIONS':
        return '', 200

    app.logger.debug("Authentication request received")
    data = request.json
    user_id = data['userId']
    image_data = data['image'].split(',')[1]
    app.logger.debug(f"User ID: {user_id}")
    app.logger.debug(f"Image data length: {len(image_data)}")

    try:
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        app.logger.debug(f"Decoded image shape: {image.shape}")

        success, message = auth_system.authenticate(user_id, image)
        app.logger.debug(f"Authentication result: success={success}, message={message}")
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        app.logger.error(f"Error during authentication: {str(e)}")
        return jsonify({'success': False, 'message': f"Error during authentication: {str(e)}"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True,host='0.0.0.0', port=port)