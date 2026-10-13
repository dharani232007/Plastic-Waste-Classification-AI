# ♻️ Deep Neural Network-Based Plastic Waste Classification

A Deep Learning computer vision application designed to automate the identification and classification of plastic waste. This project uses a custom-trained YOLOv8 model to detect mixed waste items in real-time and maps them to their specific plastic polymer types (PET, HDPE, PP) to streamline industrial recycling processes.

## 🚀 Key Features

*   **Real-Time Object Detection**: Identifies multiple plastic waste items (bottles, caps, cups, straws, etc.) within a single image.
*   **Polymer Resin Mapping**: Automatically translates recognized objects into their specific recyclable plastic types (e.g., categorizing a "Bottle" as PET/Type 1 and a "Bottle cap" as HDPE/Type 2).
*   **High-Contrast Dashboard**: Features a professional, dashboard-style Web UI built with Gradio Blocks and custom CSS for clear visual analysis.
*   **Edge-Case Handling**: Includes robust error handling for empty uploads and low-confidence visual noise.

## 🛠️ Technology Stack

*   **Core AI/Model**: YOLOv8 (Ultralytics) trained for 100 epochs
*   **Dataset**: TACO (Trash Annotations in Context)
*   **Web Framework**: Gradio
*   **Image Processing**: OpenCV, NumPy
*   **Language**: Python 3

## 🧠 How It Works

1.  **Image Input**: The user uploads an image of mixed waste to the web interface.
2.  **Inference**: The YOLOv8 neural network processes the image, generating bounding boxes and confidence scores for recognized objects (minimum confidence threshold set to 20% to account for transparent plastics).
3.  **Classification Logic**: The detected object classes are parsed through a custom mapping dictionary to identify the chemical composition (resin code) of the plastic.
4.  **Output**: The system returns an annotated image and a generated Recycling Analysis Report.

## 💻 Local Installation & Setup

Follow these steps to run the project locally on your machine.

**1. Clone the repository**
```bash
git clone [https://github.com/dharani232007/Plastic-Waste-Classification-AI.git](https://github.com/dharani232007/Plastic-Waste-Classification-AI.git)
cd Plastic-Waste-Classification-AI


Markdown
**2. Install dependencies**
Make sure you have Python installed, then run:
```bash
pip install ultralytics gradio opencv-python numpy
3. Add the Model
Ensure that your trained YOLOv8 weights file (best.pt) is located in the root directory of the project.

4. Run the Application

Bash
python app.py
The application will launch locally at http://127.0.0.1:7860.

📈 Future Scope
Expanded Dataset: Train the model on a larger dataset of highly compressed and crushed plastics often found in commercial recycling facilities.

Extended Epochs: Retrain the model for 300+ epochs to further improve detection accuracy on highly reflective and transparent surfaces.

Hardware Integration: Connect the detection pipeline to a live camera feed over a robotic sorting conveyor belt.

👨‍💻 Author
Dharani

GitHub: @dharani232007

Location: Coimbatore, Tamil Nadu
