# Sign Language Alphabet Detection Project

## Project Overview
This project involves real-time detection of sign language through a webcam and outputting the detected signs as text. Using YOLO for hand detection and LSTM for temporal sequence learning, this system translates sign language into text in real-time. The project is implemented using Spring Boot and Flask for the web interface and server communication.

## Features

1. **Webcam Sign Language Detection**: 
   - The webcam captures sign language gestures in real-time.
   - YOLO detects the hands, and Mediapipe extracts hand landmarks.
   - The LSTM model processes the temporal sequence of gestures.
   - The detected signs are translated into text and displayed on the web interface.

2. **Real-time Text Output**: 
   - The recognized sign language gestures are converted to text on the screen.
   - Real-time processing ensures immediate feedback for the user.

3. **Integration of Flask and Tomcat**: 
   - Python’s Flask server processes image data and predictions.
   - Spring Boot (Tomcat) handles the front-end display and real-time webcam streaming.

## Technologies & Tools

- **Languages**: Java (Spring Boot), Python (Flask), HTML, CSS, JavaScript
- **Frameworks**: Spring Boot 3.3.2, Flask, Thymeleaf
- **Machine Learning Models**: YOLO (You Only Look Once) for object detection, LSTM for sequence prediction
- **Computer Vision**: Mediapipe for hand landmark detection
- **Development Tools**: Pycharm (Python), Eclipse (Java), Tomcat (Web server)
  
## Key Technologies and Their Roles

1. **YOLO (You Only Look Once)**  
   - YOLO is used for real-time hand detection from the webcam stream. It’s fast and capable of detecting objects efficiently, making it suitable for hand gesture detection.

2. **LSTM (Long Short-Term Memory)**  
   - LSTM handles the temporal sequence of hand movements and interprets gestures based on previous time steps. This is crucial for detecting dynamic signs such as letters like "J" and "Z," which require motion.

3. **Mediapipe**  
   - Mediapipe is used to extract hand landmarks, which are processed to create feature vectors for model prediction.

4. **Spring Boot & Flask Integration**  
   - Flask serves as the back-end model prediction API, processing the image data and returning predictions.
   - Spring Boot manages the front-end web interface, handles real-time webcam streaming, and communicates with the Flask server.

## System Configuration

1. **Web Interface**: 
   - A web page is built using Spring Boot and Thymeleaf to capture webcam input and display text output.
   - The webcam stream is sent to the Flask server for processing.
   
2. **Flask Server**: 
   - The Flask server receives the images from the webcam, processes them using the YOLO and LSTM models, and returns the predicted sign language as text.
   
3. **Spring Boot Application**: 
   - The Spring Boot application integrates the Flask server, sends image data for predictions, and displays the output text in real-time.

## Data Collection & Processing

- **Data Collection**: 
   - Sign language data is collected using a webcam, and YOLO detects hands in each frame.
- **Preprocessing**:
   - Hand landmarks are extracted using Mediapipe.
   - Feature vectors (distances and angles between hand joints) are calculated for LSTM model input.
   - Processed data is stored in NumPy format (`.npy` files) for training.

## Demo Video

You can watch the demo of this project on YouTube by clicking the link below:  
[Hotel Reservation System Demo](https://youtube.com/your-demo-link)
