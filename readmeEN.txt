# YOLO Real-Time Object Detection and Tracking
A simple project that performs real-time object detection and tracking using YOLO on a webcam feed.

Bounding boxes are drawn around objects, and the object name, confidence score, and tracking ID are displayed. Additionally, the FPS value is shown on the screen.

The project includes two different versions for CPU and NVIDIA GPU.

CPUrun:
* Uses the yolo11n model.
* Runs on the CPU.
* Tested at approximately 13 FPS.

CUDAGPUrun:
* Uses the yolo11m model.
* Runs on an NVIDIA GPU with CUDA.
* Tested at approximately 26 FPS.
* Includes colors based on object IDs and an object counter.

## Installation

In the CPUrun folder:

cd CPUrun
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

For CUDAGPUrun:

cd CUDAGPUrun
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

## Running
In the respective folder:

venv\Scripts\activate
python main.py

After the program opens, you can press the Q key to exit.

Model files are downloaded automatically on the first run.

## Settings
Camera number, resolution, confidence value, model, and tracking algorithm can be changed from the config.json file.

CUDAGPUrun requires an NVIDIA GPU and CUDA-supported PyTorch.

## Contact
Atakan Koçoğlu - atakankocoglu.iletisim@gmail.com
