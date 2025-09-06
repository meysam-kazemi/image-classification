# MNIST Classifier: Custom CNN vs. Transfer Learning in PyTorch

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/meysam-kazemi/image-classification/blob/main/LICENSE)

An educational project that builds, trains, and compares two deep learning models for classifying the famous MNIST handwritten digit dataset. This repository serves as a practical guide to understanding the trade-offs between building a model from scratch versus using transfer learning.

![vid](https://github.com/meysam-kazemi/image-classification/blob/main/video/vid.gif)


## 📜 Project Overview

This project implements two distinct approaches to image classification using PyTorch:

1.  **Custom Convolutional Neural Network (CNN)**: A simple, lightweight CNN designed specifically for the 28x28 grayscale MNIST images.
2.  **Transfer Learning Model**: Utilizes a pre-trained `MobileNetV2` model, originally trained on the ImageNet dataset, and adapts it for the MNIST task.

The primary goal is to compare their performance, architecture, and the number of trainable parameters to highlight when each approach is most effective.

***

## 📂 Project Structure

The codebase is organized into a clean and modular structure to promote readability and maintainability.

```bash
mnist_classifier/
├── models/         # Stores the trained model weights (.pth files)
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Handles loading and transforming MNIST data
│   ├── cnn_model.py      # Defines the custom CNN architecture
│   ├── transfer_model.py # Defines the transfer learning model architecture
│   └── train.py          # Contains the training and evaluation logic
├── .gitignore            # Specifies files to be ignored by Git
├── app.py                # The Gradio web application for live testing
├── main.py               # Main script to run the training and comparison
├── requirements.txt      # Lists all project dependencies
└── README.md             # You are here!
```

***

## 🛠️ Technologies Used

* **Python 3.9+**
* **PyTorch**: The core deep learning framework.
* **Torchvision**: For accessing datasets (MNIST) and pre-trained models (MobileNetV2).
* **NumPy**: For numerical operations.

***

## ⚙️ Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.9** or later installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/meysam-kazemi/image-classification
cd image-classification
```

### 3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
Install all the required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

***

## ▶️ How to Run

### Train the Models
To start the training process for both models and see the final comparison, simply run the main script:
```bash
python main.py
```
The script will automatically use a CUDA-enabled GPU if one is available; otherwise, it will fall back to the CPU. The trained models will be saved in the `saved_models/` directory.

### Launch the Interactive Demo
Once the models have been trained, you can launch the Gradio web app to test them.

```bash
python app.py
```

Open the local URL provided in your terminal to start drawing and classifying digits.



***

## 📊 Results and Comparison

After running the script, a comparison of the two models will be printed to the console. The expected results are summarized below:

| Metric | Custom CNN | Transfer Learning (MobileNetV2) |
| :--- | :---: | :---: |
| **Test Accuracy** | **99.28%** | 66.52% |
| **Trainable Parameters**| ~225,000 | ~165,000 |
| **Total Parameters** | ~225,000 | ~2,390,000 |

### Analysis 💡

* **Why does the custom CNN perform better?** The MNIST dataset is relatively simple and large. Our custom CNN is built specifically for this task, allowing it to learn highly relevant features for digit recognition. The features learned by `MobileNetV2` from the complex ImageNet dataset (e.g., textures, animal shapes) are overly powerful and not a perfect match for identifying simple digits, leading to slightly lower performance.

* **When is Transfer Learning better?** Transfer learning is incredibly powerful when you have a **small but complex dataset**. For example, classifying 1,000 images of different flower species. Training a deep model from scratch would be impossible, but by using a pre-trained model, you leverage its robust feature extraction capabilities and can achieve high accuracy with minimal data.

This project clearly demonstrates that the "best" model architecture is always dependent on the specific problem you are trying to solve!

***

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, feel free to create an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
