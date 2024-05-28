 # Real-Time-Face-Recognition
This repository is dedicated to the Project "End-to-End real-time face recognition system" 

## Usage

### Install Dependencies
* Clone the Project Repository to your system
```bash
git clone <GITHUB_REPO_URL>
```
* Go to the Project Directory and Create a Virtual Environment in Python

```bash
python -m venv face-recognition-system
source face-recognition-system/bin/activate

```

```py
pip install requirements.txt
```

### Start the Flask API Server

* Navigate to src/api Folder
```bash
python endpoints.py
```
**Note:** This will start your Flask API at : **http://localhost:5000/**  
The API has two endpoints (you can test the API using POSTMAN)
>* **http://localhost:5000/api/register**(for User FaceID registration)
>* **http://localhost:5000/api/recognize** (for recognizing user given face image)

### Start the Interactive UI(Streamlit)
* After Starting the API, We can use the Face Recognition system interactively through streamlit application

* Run the command
```bash
streamlit run streamlit_app.py
```

* Streamlit UI will start at : **http://localhost:8501**
