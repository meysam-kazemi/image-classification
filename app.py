import gradio as gr
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL.ImageOps
from configparser import ConfigParser

# Import your model class definitions from the src folder
from src.cnn_model import CNNModel
from src.transfer_model import build_transfer_model

config = ConfigParser()
config.read('config.ini')

# --- 1. SETUP ---
# Determine the device to run the models on (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the custom CNN model
cnn_model = CNNModel()
# Load the saved weights; map_location ensures it works even if you trained on a GPU and are now on a CPU
cnn_model.load_state_dict(torch.load(config.get('MODEL', 'cnn_path'), map_location=device))
cnn_model.to(device)
cnn_model.eval() # Set the model to evaluation mode

# --- 2. DEFINE PREPROCESSING AND PREDICTION ---
# Define the image transformations for each model
# These must be the SAME as the ones used during training
cnn_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# The main prediction function that Gradio will call
def predict(drawing):
    if drawing is None:
        return "Please draw a digit first!"

    # The drawing is a NumPy array. Convert it to a PIL Image for easy transformation.

    img = Image.fromarray(drawing['composite'].astype('uint8'), 'L')
    img = PIL.ImageOps.invert(img)

    # Select the model and the corresponding transformation pipeline
    model = cnn_model
    transformed_img = cnn_transform(img)

    # Add a batch dimension (models expect inputs in batches)
    # The shape should be [1, C, H, W]
    batch = transformed_img.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(batch)
    
    # Get probabilities by applying softmax to the logits
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Create a dictionary of labels and their probabilities
    # Gradio's Label component can display this nicely
    confidences = {str(i): float(prob) for i, prob in enumerate(probabilities)}
    
    return confidences

# --- 3. CREATE THE GRADIO INTERFACE ---
# Define the input and output components for the web app
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Sketchpad(
            image_mode="L", 
            #invert_colors=True,
            label="Draw a Digit (0-9)",
            width=750,
            height=750
        ),
    ],
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    live=False,
    title="MNIST Digit Recognizer",
    description="Draw a digit on the canvas"
)

# --- 4. LAUNCH THE APP ---
if __name__ == "__main__":
    iface.launch()
