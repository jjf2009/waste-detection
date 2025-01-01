# **AI Anti-Garbage Disposal System**

## CTI Practical AEC-152

### **Overview**
The AI Anti-Garbage Disposal System is designed to tackle illegal garbage dumping and promote responsible waste management. This project leverages AI-driven technology, including motion detection, object detection, and facial mapping, to monitor areas prone to illegal dumping. The ultimate objective is to reduce environmental pollution, foster proper waste disposal habits, and ensure cleaner surroundings.

Currently, the system supports motion detection, object detection, and facial recognition using a camera feed. Future enhancements aim to integrate more advanced AI capabilities, such as automated reporting to authorities and improved classification of waste types.

---

### **Setup and Installation**
#### **1. Download and Extract the Project**
- Click "Code" above and select **Download ZIP**.  
  Alternatively, use **GitHub Desktop** to clone the repository.  
- Extract the contents to your desired folder.  

#### **2. Prerequisites**
- **Python Version**: Ensure Python 3.9 is installed. Python 3.10 and above are not supported due to compatibility issues with certain libraries (e.g., Mediapipe).  
- **Compiler Tools**: Install **Microsoft Visual C++ Compiler for Python**.  
- Verify Python's PATH is set correctly in the system's Environment Variables.  

#### **3. Install Required Components**
1. Open Command Prompt and navigate to the project folder.  
2. Create and activate a virtual environment:  
   ```bash
   py -3.9 -m venv myenv39
   myenv39\Scripts\activate
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
   If you encounter errors, install them manually:  
   ```bash
   pip install opencv-python imutils ultralytics mediapipe onnx
   ```  

4. Confirm installation by running:  
   ```bash
   pip list
   ```  

---

### **Running the Program**
#### **1. Set Up the Model**
- Ensure the YOLO model file (`best.pt`) is in the directory specified in `settings.py`.  

#### **2. Run the Application**
- Navigate to the directory containing `AI_Anti_Gabrage.py` and execute:  
  ```bash
  python AI_Anti_Gabrage.py
  ```  

#### **3. Optional: Train the YOLO Model**
- If retraining is required, navigate to the directory containing `train.py` and run:  
  ```bash
  python train.py
  ```  

---

### **File Overview**
#### **1. AI_Anti_Gabrage.py**
This is the main script for real-time pose and object detection using a webcam.  
- **Features**:  
  - Pose detection using Mediapipe.  
  - Object detection and classification using YOLO.  
  - Waste categorization: Recyclable, non-recyclable, and hazardous.  
- **Key Functions**:  
  - Display landmarks, detected objects, and waste classification in the video feed.  

#### **2. motiondetector.py**
Detects motion in video feeds using OpenCV.  
- **Features**:  
  - Captures video from a file or webcam.  
  - Highlights motion by drawing bounding boxes.  

#### **3. settings.py**
Configuration file containing paths, waste category definitions, and other global settings.  

#### **4. train.py**
Script for training the YOLO model on custom datasets.  
- **Features**:  
  - Initializes and trains the YOLO model using a dataset.  
  - Exports the trained model for use in object detection.  

---

### **How Components Are Linked**
- **AI_Anti_Gabrage.py**:  
  - Uses configurations and waste categories from `settings.py`.  
  - Loads and uses the trained YOLO model for object detection.  
- **settings.py**:  
  - Provides the required paths and waste classifications.  
- **train.py**:  
  - Produces a trained YOLO model used by `AI_Anti_Gabrage.py`.  
- **motiondetector.py**:  
  - A standalone module for motion detection that can be integrated if required.  

---

### **Dependencies**
The following Python libraries are used in the project:  
- **OpenCV** (`opencv-python`): Real-time computer vision tasks.  
- **Mediapipe**: Pose and landmark detection.  
- **NumPy**: Numerical computations.  
- **Ultralytics**: YOLO model handling.  
- **Imutils**: Simplifies OpenCV tasks.  
- **ONNX**: Exporting the trained model.  

---

### **Future Scope**
- Enhance the object classification system for more accurate waste categorization.  
- Integrate automated reporting features to notify authorities.  
- Add support for edge devices for real-time monitoring in remote areas.  

---

