import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, VideoTransformerBase
import numpy as np
import av
from PIL import Image
from endpoints import recognize_face, register_user
import requests
import io
import tempfile
import time

st.title("Real Time Face Recognition System Live Demo")


# Page for user registration form
def registration_page():
    st.title('User FaceID Registration')
    name = st.text_input('Name:')
    email = st.text_input('Email:')
    capture_option = st.radio('Image Source:', ('Upload Image', 'Capture Image'))

    if capture_option == 'Upload Image':
        uploaded_image = st.file_uploader('Upload Image:', type=['jpg', 'png'])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
             # Convert the image to bytes
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='JPEG')
            byte_arr.seek(0)
            jpg_as_text = byte_arr.read()
            if st.button('Register', key='register_button'):
                data = {
                    'name': name,
                    'email': email,
                }
                response = requests.post('http://localhost:5000/api/register', data = data, files={'image': ('image.jpg', jpg_as_text, 'image/jpeg')})
                result = response.json()
                if 'error' in result:
                    st.error('Registration failed:' + result['error'])
                else:
                    st.success('Registration Successful!')
                ## CHeck if this API Call is making the user register in the database
                if response.status_code == 200:
                    st.success('Registration successful!')
                else:
                    st.error('Registration failed!')

    # elif capture_option == "Capture Image":
            
    #         FRAME_WINDOW = st.image([])
    #         camera = cv2.VideoCapture(0)
    #         image= None
    #         run = st.button('Capture Image',  key='capture_button')
    #         while run:
    #             _, frame = camera.read()
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             FRAME_WINDOW.image(frame)
    #             image = Image.fromarray(frame.astype('uint8'))
    #             byte_arr = io.BytesIO()
    #             image.save(byte_arr, format='JPEG')
    #             byte_arr.seek(0)
    #             run = False
    #             # print(type(byte_arr))
    #             # print(type(image))
    #         if image is not None:
    #             st.image(image, caption= 'Captured Image' , use_column_width=True)
    #             if st.button('Register', key='register_button'):
    #                 print("register button entered")
    #                 byte_arr.seek(0)
    #                 jpg_as_text = byte_arr.read()
    #                 print("jpg as text done")
    #                 data = {
    #                     'name': name,
    #                     'email': email,
    #                 }
    #                 print("data loaded as json")
    #                 response = requests.post('http://localhost:5000/api/register', data = data, files={'image': ('image.jpg', jpg_as_text, 'image/jpeg')})
    #                 print('response is requested')
    #                 result = response.json
    #                 if 'error' in result:
    #                     st.error('Registration failed:' + result['error'])
    #                 else:
    #                     st.success('Registration Successful!')
    #                 ## CHeck if this API Call is making the user register in the database
    #                 if response.status_code == 200:
    #                     st.success('Registration successful!')
    #                 else:
    #                     st.error('Registration failed!')
            # st.image(image, caption= 'Captured Image' , use_column_width=True)
                
                # if st.button('Capture Image'):
                #     is_success, buffer = cv2.imencode(".jpg", frame)
                #     if is_success:
                #         st.image(buffer, caption='Captured Image', use_column_width=True)
                #         # jpg_as_text = buffer.tobytes()
                        

            # frame = capture_faces_test() # Replace with capture image function
            # image = Image.fromarray(frame)
            # image_bytes = image.tobytes()
            # st.image(image, caption='Captured Image', use_column_width=True)

    # if st.button('Register', key='register_button'):
    #     # Perform registration process here
    #     data = {
    #         'name': name,
    #         'email': email,
    #     }
    #     files = {
    #     'image': ('image.jpg', 'image/jpeg')
    #     }

    #     response = requests.post('http://localhost:5000/api/register', data = data, files=files)

    #     ## CHeck if this API Call is making the user register in the database
    #     if response.status_code == 200:
    #         st.success('Registration successful!')
    #     else:
    #         st.error('Registration failed!')


# Page for user login using face image
def login_page():
    st.title('User Login with Face ID')
    st.write('Use the camera feed below to capture your face image for login.')
   
    run = st.button("Open Web Cam", key = "open_web_cam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    frame = None
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(type(frame))
        FRAME_WINDOW.image(frame)
        run = False
    if st.button('Enter', key = 'enter_button'):

        # Convert the frame to a PIL Image
        image = Image.fromarray(frame)

        # Create a BytesIO object and save the image in JPEG format to it
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='JPEG')

        # Get the byte array of the JPEG image
        jpg_as_text = byte_arr.getvalue()

        # Send the image data to the API
        response = requests.post('http://localhost:5000/api/recognize', files={'image': ('image.jpg', jpg_as_text, 'image/jpeg')})
        result = response.json()
        username = result['username']
        if username is not None:
            return user_dashboard(username)
        else:
            return st.error('Login failed!')

        # Check the response
        if response.status_code == 200:
            st.write('Image sent successfully')
        else:
            st.write('Failed to send image')



def user_dashboard(username):
    st.title('User Dashboard')
    st.title(f"Welcome Back!!", {username})
    st.write('User Details')

    # Connect to the MySQL database and fetch the user data
    cnx = mysql.connector.connect(user='mysql_user', password='mysql_password',
                                  host='127.0.0.1',
                                  database='my_database')
    cursor = cnx.cursor()

    query = ("SELECT username, email, face_embedding FROM users "
             "WHERE username = %s")
    cursor.execute(query, (username,))

    for (username, email, face_embedding) in cursor:
        st.write(f'Username: {username}')
        st.write(f'Email: {email}')

       ## TODO: Display user image here

    cursor.close()
    cnx.close()



# App navigation
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('', ('User Registration', 'User Login'))
    if page == 'User Registration':
        registration_page()
    elif page == 'User Login':
        login_page()


if __name__ == '__main__':
    main()




 
    # # image_capture = ImageCapture()
    # img = webrtc_streamer(
    #     key="streamer",
    #     video_frame_callback=transform,
    #     sendback_audio=False
    # )
    # Capture image
    # captured_image = webrtc_streamer(
    #     key="example",
    #     video_processor_factory=None,  # We don't need video processing, just capturing the frame
    #     desired_output_format=st.StreamlitRenderOutput(format="JPEG")  # Capture image in JPEG format
    # )



# from PIL import Image
# from io import BytesIO

# # Assume you have a streamlit_webrtc_streamer setup here

# # Function to send image to Flask API
# def send_image_to_flask(image):
#     url = 'http://localhost:5000/recognize_face/'  # URL of your Flask API

#     # Convert image to bytes
#     img_byte_array = BytesIO()
#     image.save(img_byte_array, format='PNG')
#     img_byte_array = img_byte_array.getvalue()

#     # Send image to Flask API
#     # files = {'file': ('image.png', img_byte_array)}
#     response = requests.post(url, data = img_byte_array)

#     return response.text

# pg_as_text = buffer.tobytes()




    # if st.button('Login'):
    #     st.title()
    #     # captured_face = capture_faces_test()
    #     image_bytes  = captured_face.tobytes()
    #     # Perform face recognition and login process here
    #     # Convert the image to bytes
    #     # _, buffer = cv2.imencode('.jpg', image_capture.frame)
    #     # image_bytes = buffer.tobytes()
        
 


    #     # Send the image to the recognize_face API endpoint
    #     # Convert the image to JPEG format
    #     # image_jpg = cv2.imencode('.jpg', image_capture.frame)[1].tobytes()
        
    #     # Send the image to the recognize_face API endpoint
    #     response = requests.post('http://localhost:5000/api/recognize', data = image_bytes)

        # Get the result from the response
        

        # # Extract the username from the result
        # username = result['username']
        # ## TODO: Send API Request to backend for face recognition and Authenticate the User with user Details
        # username = recognize_face(img)
        # st.success('Login successful!')
        # ## TODO: Redirect to User Details Page
        # if username is not None:
        #     return user_dashboard(username)
        # else:
        #     return st.error('Login failed!')



############################## Image Capture Callbacks ##############################


# class ImageCapture(VideoTransformerBase):
#     def __init__(self):
#         self.frame = None
    
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         return frame

# def transform(frame: av.VideoFrame) -> np.ndarray:
#     img = frame.to_ndarray(format="bgr24")
#     return img

# # Function to capture image from webcam
# def capture_image():
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#     cap.release()
#     return frame

# def capture_face():
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # Convert frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         # Draw rectangles around detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Display the frame
#         cv2.imshow('frame', frame)

#         # Check if a face is straight in the frame and capture it
#         if len(faces) > 0:
#             x, y, w, h = faces[0]  # Assuming only one face is detected
#             if w > 100 and h > 100:  # Adjust this condition as needed
#                 # Crop the face region
#                 face_region = frame[y:y+h, x:x+w]

#                 # Convert the captured face into numpy array
#                 numpy_array_face = np.asarray(face_region)

#                 # Print the numpy array
#                 print("Captured face as numpy array:")
#                 print(numpy_array_face)

#                 # Release the webcam
#                 cap.release()
#                 cv2.destroyAllWindows()

#                 return numpy_array_face

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


#     # Release the webcam
#     cap.release()
#     cv2.destroyAllWindows()

# def capture_faces_test():
#     # Initialize the webcam
#     cap = cv2.VideoCapture(0)
    
#     # Check if the webcam is opened correctly
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             continue

#         # Display the frame
#         cv2.imshow('frame', frame)

#         # Capture key press
#         key = cv2.waitKey(1) & 0xFF

#         # Check if 'c' key is pressed to capture the frame
#         if key == ord('c'):
#             print("Key 'c' pressed.")
#             # Convert the captured frame into numpy array
#             numpy_array_frame = np.asarray(frame)

#             # Print the numpy array shape and content
#             print("Captured frame as numpy array with shape:", numpy_array_frame.shape)
#             print(numpy_array_frame)

#             # Optionally, save the captured frame as an image
#             cv2.imwrite('captured_frame.jpg', frame)

#             # Break the loop to shut down the webcam
#             break

#         # Break the loop if 'q' is pressed
#         elif key == ord('q'):
#             print("Key 'q' pressed. Exiting.")
#             break

#     # Release the webcam
#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage