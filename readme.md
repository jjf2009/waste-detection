<ins>#**AI Anti Garbage Disposal System**</ins>\
**##CTI Practical AEC-152**
\
\

##Overview
This project focuses on developing an AI-driven anti-garbage disposal system to combat illegal dumping and promote responsible waste management in residential areas. 
Initially, the proposed plan was to have the system use a AI-powered camera and specific machine learning algorithms to detect and record instances of illegal dumping. This information would then be transmitted to law enforcement authorities, for them to take further action. 
The end goal is to reduce environmental pollution by preventing garbage accumulation, encouraging proper waste disposal habits, and ultimately creating a cleaner environment.\
For now, we have developed a system that detects motion of bodies, detects objects and identifies/maps faces that appear on the camera feed.
<br/>
<br/>
##Instructions to run this Project
###Installation
1. Downloading and Extracting
    a. Click on "Code" above, under the tab, click on "Download ZIP".<br/>
       _Otherwise, click on "Github Desktop" and download the project._
    b. Extract the ZIP contents to your desired folder.
2. Prerequisites and Installing components
> [!IMPORTANT]
> Ensure you have atleast Python 3.9 and Microsoft Visual C++ Compiler for Python Installed on your system. Also ensure the PATH location under Environment Variables (Windows) is properly linked to Python 3.9 folder. Certain imported files and features do not run on the old, as well as latest versions of Python (Python 3.10 will not work as mediapipe does not support it).
    a. Install the following build tools - gcc, make, wget via the command prompt terminal on Windows system.
    b. Open a new instance of command line interface/ command prompt, and navigate to the extracted folder.
    c. Create a Virtual Environment: Open Command Prompt and create a virtual environment:
        `py -3.9 -m venv myenv39`
    d. Activate the Virtual Environment: Activate your new environment:
        `py -3.9 -m venv myenv39`
    e. Install Required Packages: Install the necessary packages/dependencies:
        `pip install -r requirements.txt`
       Or, if you encounter any errors, you can download the dependencies manually:
        ```
        pip3 install opencv-python
        pip install imutils
        pip install ultralytics
        pip install mediapipe
        pip install onnx
        ```
        After installing, run `pip list` and you must see the following:
        ```
        absl-py               2.1.0
        attrs                 24.3.0
        certifi               2024.12.14
        cffi                  1.17.1
        charset-normalizer    3.4.1
        colorama              0.4.6
        contourpy             1.3.0
        cycler                0.12.1
        filelock              3.16.1
        flatbuffers           24.12.23
        fonttools             4.55.3
        fsspec                2024.12.0
        idna                  3.10
        importlib_metadata    8.5.0
        importlib_resources   6.4.5
        imutils               0.5.4
        jax                   0.4.30
        jaxlib                0.4.30
        Jinja2                3.1.5
        kiwisolver            1.4.7
        MarkupSafe            3.0.2
        matplotlib            3.9.4
        mediapipe             0.10.20
        ml_dtypes             0.5.0
        mpmath                1.3.0
        networkx              3.2.1
        numpy                 1.26.4
        onnx                  1.17.0
        opencv-contrib-python 4.10.0.84
        opencv-python         4.10.0.84
        opt_einsum            3.4.0
        packaging             24.2
        pandas                2.2.3
        pillow                11.0.0
        pip                   24.3.1
        protobuf              4.25.5
        psutil                6.1.1
        py-cpuinfo            9.0.0
        pycparser             2.22
        pyparsing             3.2.0
        python-dateutil       2.9.0.post0
        pytz                  2024.2
        PyYAML                6.0.2
        requests              2.32.3
        scipy                 1.13.1
        seaborn               0.13.2
        sentencepiece         0.2.0
        setuptools            49.2.1
        six                   1.17.0
        sounddevice           0.5.1
        sympy                 1.13.1
        torch                 2.5.1
        torchvision           0.20.1
        tqdm                  4.67.1
        typing_extensions     4.12.2
        tzdata                2024.2
        ultralytics           8.3.55
        ultralytics-thop      2.0.13
        urllib3               2.3.0
        zipp                  3.21.0
        ```
###Running the Program
3. Prepare and Run
    a. Prepare the YOLO Model: <br/>
       Ensure the YOLO model file (best.pt) is in the correct directory as specified in settings.py.
    b. Run the Main Script:<br/>
       Navigate to the directory containing AI_Anti_Gabrage.py and run it:
       `python AI_Anti_Gabrage.py`
    c. Train the YOLO Model (if needed): <br/>
       Navigate to the directory containing train.py and run it:
       `python train.py`
      <br/>
      <br/>
##What each file does
1. AI_Anti_Gabrage.py
   This script is the main application for detecting poses and objects in real-time using a webcam. It uses Mediapipe for pose detection and YOLO for object detection.<br/>
    Imports: Imports necessary libraries like OpenCV, Mediapipe, NumPy, and YOLO.<br/>
    Global Variables: Initializes global variables for counting and stage tracking.<br/>
    Video Capture: Initializes video capture from the webcam.<br/>
    Model Loading: Loads the YOLO model for object detection.<br/>
    Pose Detection: Uses Mediapipe to detect human poses and calculate angles.<br/>
    Object Detection: Uses YOLO to detect objects in the video frame.<br/>
    Waste Classification: Classifies detected objects into recyclable, non-recyclable, and hazardous categories.<br/>
    Display: Displays the video feed with pose landmarks, detected objects, and waste classification.<br/>

2. motiondetector.py
   This script detects motion in a video feed using OpenCV.<br/>
    Imports: Imports necessary libraries like OpenCV and imutils.<br/>
    Argument Parsing: Parses command-line arguments for video file path and minimum area size.<br/>
    Video Capture: Captures video from a file or webcam.<br/>
    Motion Detection: Compares the current frame with the first frame to detect motion.<br/>
    Display: Draws bounding boxes around detected motion and displays the video feed.<br/>

3. settings.py
   This script contains configuration settings for the project.<br/>
    Paths: Sets up paths for the model and webcam.<br/>
    Waste Categories: Defines lists of recyclable, non-recyclable, and hazardous waste items.<br/>

4. train.py
   This script trains the YOLO model for waste detection.<br/>
    Model Initialization: Initializes the YOLO model with a configuration file.<br/>
    Training: Trains the model using a dataset specified in a YAML file.<br/>
    Validation: Validates the trained model.<br/>
    Export: Exports the trained model in ONNX format.<br/>
<br/>
<br/>
##How all the components and files are linked
1. AI_Anti_Gabrage.py:<br/>
    Imports settings from settings.py: This file uses configurations defined in settings.py such as the model path and waste categories.<br/>
    Loads the YOLO model: It uses the model path specified in settings.py to load the YOLO model.<br/>
    Performs pose detection and object detection: It uses Mediapipe for pose detection and YOLO for object detection.<br/>
   
2. settings.py:<br/>
    Provides configuration settings: This file defines paths and waste categories used by AI_Anti_Gabrage.py.<br/>

3. train.py:<br/>
    Trains the YOLO model: This file is used to train the YOLO model. Once trained, the model can be used by AI_Anti_Gabrage.py for object detection.
  
5. motiondetector.py:
    Detects motion: This file is separate and used for motion detection. It is not directly linked to AI_Anti_Gabrage.py but can be used independently if motion detection is required.

##Dependencies
The following dependencies were used (not included in original Python)

opencv-python                By OpenCV Org
mediapipe                    Google AI Edge
numpy                        Numpy
ultralytics                  YOLO
imutils                      PyImageSearch

All credits go to these developer organisations for their respective dependencies and packages.
