from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from face_recognition_model import FaceRecognizer
from preprocessing import Preprocessor
import mysql.connector
import cv2
import numpy as np  
from PIL import Image
import io
import os


app = Flask(__name__)

# Database Related configs:

db_host = 'db-mysql-cluster-trimx-do-user-15662268-0.c.db.ondigitalocean.com'
db_user = 'doadmin'
db_pass = 'passwd' # Enter Password here
db_name = 'test'
db_port = '25060'



@app.route('/api/recognize', methods=['POST'])
def recognize_face():

    
    if 'image' not in request.files:        
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
     # Check the file extension
    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid image format'}), 400
    
    image = Image.open(image_file)
    img = np.array(image)

    preprocessor = Preprocessor(None, 160, 160, 3)

    script_dir = os.path.dirname(__file__) # get the directory of the current script
    src_dir= os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, 'models/weights/facenet.tflite')
    recognizer = FaceRecognizer(model_path)
    img = preprocessor.face_detect(img) # Detect face
    img = preprocessor.face_align(img) # Align Face
    img = preprocessor.crop_face_image(img) # Crop face image

    embedding = recognizer.get_embedding(img).tolist()
    embedding = recognizer.get_embedding(img)

    # Connect to the MySQL database and fetch the user embeddings
    cnx = mysql.connector.connect(user=db_user, password=db_pass,
                                  host=db_host,
                                  database=db_name, port = db_port)
    cursor = cnx.cursor()

    query = ("SELECT username, face_embedding FROM users")
    cursor.execute(query)
    threshold = 0.9
    min_distance = float('inf')
    closest_username = None
    for (username, face_embedding) in cursor:
        face_embedding = face_embedding.strip('[]')
        face_embedding = face_embedding.replace(' ', ',')
        face_embedding = np.fromstring(face_embedding, dtype=float, sep=',')
        distance = np.linalg.norm(embedding - face_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_username = username
    print(min_distance)
    if min_distance > threshold:
        print(min_distance)
        closest_username = None
    cursor.close()
    cnx.close()

    return jsonify({'username': closest_username})




@app.route('/api/register', methods=['POST'])
def register_user():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Check if the post request has the required fields
    if 'username' not in request.form or 'email' not in request.form:
        return jsonify({'error': 'No username or email provided'}), 400


    username = request.form['username']
    email = request.form['email']

    image_ = request.files['image']
    image = Image.open(image_.stream)
    image_array = np.array(image)
    # Process the image and get the embedding
    preprocessor = Preprocessor(None, 160, 160, 3)
    script_dir = os.path.dirname(__file__) # get the directory of the current script
    src_dir= os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, 'models/weights/facenet.tflite')
    recognizer = FaceRecognizer(model_path)
    

    img = preprocessor.face_detect(image_array)  # Detect face
    img = preprocessor.face_align(img)  # Align face
    img = preprocessor.crop_face_image(img)  # Crop face image

    embedding = recognizer.get_embedding(img)  # Get the embedding of the face

    # Convert the embedding to a string to store in the database
    embedding_str = ','.join(map(str, embedding.flatten()))

    # Connect to the MySQL database and insert the user data
    cnx = mysql.connector.connect(user=db_user, password=db_pass,
                                  host=db_host,
                                  database=db_name, port = db_port)

    cursor = cnx.cursor()

    add_user = ("INSERT INTO users "
                "(username, email, face_embedding) "
                "VALUES (%s, %s, %s)")
    user_data = (username, email, embedding_str)

    cursor.execute(add_user, user_data)
    cnx.commit()

    cursor.close()
    cnx.close()

    return jsonify({'message': 'User registered successfully'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
