# Plant Disease Recognition System

## Overview
The **Plant Disease Recognition System** is a web-based application designed to help farmers, gardeners, and agricultural enthusiasts identify plant diseases quickly and accurately. By uploading an image of a plant leaf, the system uses a deep learning model to detect and classify diseases across various crops. Built with Streamlit and TensorFlow, this project aims to promote healthier crops and sustainable agriculture through accessible technology.

## Features
- **Image-Based Disease Detection**: Upload a plant leaf image to detect diseases in seconds.
- **User-Friendly Interface**: A simple, intuitive dashboard with Home, About, and Disease Recognition pages.
- **High Accuracy**: Utilizes a convolutional neural network (CNN) trained on a large dataset for reliable predictions.
- **Multi-Crop Support**: Identifies diseases across 38 classes, covering crops like apples, tomatoes, corn, grapes, and more.
- **Fast Results**: Processes images efficiently, providing instant feedback.
- **Informative Output**: Displays the predicted disease name with a clean, visual presentation.

## Dataset
The system is powered by the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle, which includes:
- **87,000+ RGB images** of healthy and diseased plant leaves.
- **38 classes** representing various crops and their diseases (e.g., Apple Scab, Tomato Late Blight, Corn Common Rust).
- **Split**: 70,295 training images, 17,572 validation images, and 33 test images.
- **Augmentation**: The dataset uses offline augmentation to enhance model robustness.

The dataset is organized into `train`, `validation`, and `test` directories, preserving the class structure for training and evaluation.

## Model Architecture
The system uses a **Convolutional Neural Network (CNN)** built with TensorFlow. The architecture includes:
- **Input Layer**: Accepts 128x128 RGB images.
- **Convolutional Layers**: Multiple Conv2D layers with increasing filters (32, 64, 128, 256, 512) for feature extraction, using ReLU activation.
- **Pooling Layers**: MaxPooling2D layers to reduce spatial dimensions and computational load.
- **Dropout Layers**: Applied (25% and 40%) to prevent overfitting.
- **Fully Connected Layers**: A dense layer with 1,500 units (ReLU) followed by a 38-unit output layer (Softmax) for classification.
- **Optimizer**: Adam with a learning rate of 0.0001.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.

The model is saved as `trained_plant_disease_model.keras` and loaded into the Streamlit app for predictions.

## Installation
To run the project locally, follow these steps:

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A modern web browser (e.g., Chrome, Firefox)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/plant-disease-recognition.git
   cd plant-disease-recognition
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   streamlit==1.31.0
   tensorflow==2.15.0
   numpy==1.26.4
   pillow==10.2.0
   ```

4. **Download the Trained Model**:
   - Place the `trained_plant_disease_model.keras` file in the project root directory. You can download it from the dataset or train it using the provided Jupyter notebook (`Plant_Disease_Detection.ipynb`).

5. **Run the Application**:
   ```bash
   streamlit run app2.py
   ```
   This will start a local server, and you can access the app in your browser at `http://localhost:8501`.

## Usage
1. **Launch the App**:
   Run the Streamlit command above to open the web interface.

2. **Navigate the Dashboard**:
   - **Home**: Learn about the system, its purpose, and how it works.
   - **About**: Explore details about the dataset and project background.
   - **Disease Recognition**: Upload an image and get disease predictions.

3. **Upload an Image**:
   - On the Disease Recognition page, click "Choose an Image" to upload a plant leaf image (JPEG or PNG).
   - Click the "Predict" button to analyze the image.

4. **View Results**:
   - The system will display the predicted disease (e.g., "Tomato___healthy" or "Apple___Black_rot").
   - If no image is uploaded, a warning will prompt you to select one.

## Training the Model
To retrain the model, use the provided Jupyter notebook (`Plant_Disease_Detection.ipynb`):
1. **Install Additional Dependencies**:
   ```bash
   pip install kagglehub matplotlib pandas seaborn
   ```
2. **Download the Dataset**:
   The notebook uses `kagglehub` to download the dataset automatically.
3. **Run the Notebook**:
   Execute the cells to load data, define the CNN, train the model, and save it as `trained_plant_disease_model.keras`.
4. **GPU Support**:
   The notebook is configured for GPU acceleration (e.g., Google Colab with T4 GPU) to speed up training.

## Project Structure
- `app2.py`: Main Streamlit application script.
- `Plant_Disease_Detection.ipynb`: Jupyter notebook for dataset loading, model training, and evaluation.
- `trained_plant_disease_model.keras`: Pre-trained model file (not included in the repository; download or train separately).
- `plant1.jpg`: Sample image displayed on the Home page.
- `requirements.txt`: List of Python dependencies.

## Contributing
We welcome contributions to improve the project! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 guidelines and includes relevant documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Dataset**: Thanks to [vipoooool](https://www.kaggle.com/vipoooool) for providing the New Plant Diseases Dataset.
- **Libraries**: Built with Streamlit, TensorFlow, and other open-source tools.
- **Community**: Inspired by the agricultural and machine learning communities working to solve real-world problems.

## Contact
For questions or feedback, feel free to open an issue on GitHub or reach out to the project maintainer at [sandeeplimbure@gmail.com].

Happy farming and disease detection! 🌱
