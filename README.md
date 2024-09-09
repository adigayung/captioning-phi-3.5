
# Image Captioning with Phi 3.5 Vision Instruct Model

This repository provides a comprehensive implementation for generating image captions using the **microsoft/Phi-3.5-vision-instruct** model. The Phi-3.5 model is designed to generate descriptive text based on the content of images, using cutting-edge deep learning techniques. Whether you're building a computer vision application, annotating datasets, or experimenting with AI, this repository will help you easily integrate image-to-text capabilities.

## Table of Contents

- [Introduction](#introduction)
- [About Phi 3.5 Vision Instruct](#about-phi-35-vision-instruct)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Introduction

The `microsoft/Phi-3.5-vision-instruct` model is a highly versatile machine learning model designed for image captioning tasks. This repository demonstrates how to use the model to generate meaningful and contextually relevant text captions based on image inputs.

This model leverages advanced instruction-based techniques, making it suitable for vision tasks that require comprehensive understanding of visual inputs. By integrating it into your workflows, you can convert images into descriptive, human-like sentences.

## About Phi 3.5 Vision Instruct

**Phi 3.5 Vision Instruct** is part of the larger "Phi" series developed by Microsoft for multimodal AI tasks. This model is particularly strong in image-to-text tasks, where its deep learning framework can analyze intricate details of an image to produce accurate, concise, and contextually appropriate descriptions.

It excels in tasks such as:
- Image Captioning
- Image Annotation for Datasets
- Visual Question Answering (VQA)
- General vision-related NLP tasks

The model has been trained on a large and diverse dataset, making it highly robust across various domains like photography, art, landscapes, and even object-based images.

## Installation

To set up the repository on your machine and start generating captions with the Phi 3.5 Vision Instruct model, follow these simple steps:

1. **Clone the Repository**:
   
   First, clone the repository to your local machine by running:

   ```bash
   git clone https://github.com/adigayung/captioning-phi-3.5.git
   ```

2. **Navigate to the Directory**:
   
   Change your working directory to the cloned folder:

   ```bash
   cd captioning-phi-3.5
   ```

3. **Grant Execution Permission for the Installation Script**:

   To ensure the installation script has the correct permissions, run:

   ```bash
   chmod +x install.sh
   ```

4. **Run the Installation Script**:

   Finally, execute the script to install all dependencies and set up the model:

   ```bash
   ./install.sh
   ```

   The installation script will handle the following tasks:
   - Install Python 3.11 if needed
   - Set up the `transformers` library and required dependencies
   - Install additional libraries such as `torch`, `Pillow`, and `requests`
   - Download the Phi 3.5 Vision Instruct model and tokenizer

## Usage

Once the installation is complete, you can use the model to generate captions for images. Here's how you can use it:

### 1. **Generate Captions for All Images in a Directory**
To generate descriptions for all images in a directory, run:

```bash
python run.py --image-dir /path/to/image-directory
```

Replace `/path/to/image-directory` with the path of the folder containing the images. The descriptions for each image will be saved in a `.txt` file with the same name as the image in the same folder.

### 2. **Generate Captions from an Image URL**
You can provide a URL of an image and get its description by running:

```bash
python run.py --url-image https://example.com/image.jpg
```

The script will fetch the image from the URL, generate its description, and print the result.

### 3. **Generate Captions for a Specific Image File**
To generate a description for a specific image file, run:

```bash
python run.py --image-file /path/to/image.jpg
```

The description will be saved as a `.txt` file in the same directory as the image.

## Model Details

The **microsoft/Phi-3.5-vision-instruct** model is pre-trained and fine-tuned for generating natural language captions from images. Some of its key features include:
- **Multimodal learning**: The model is trained to handle both visual and textual data, combining image understanding with natural language generation.
- **Instruction-based architecture**: By following given instructions, it can generalize across a wide variety of tasks such as generating captions, answering visual questions, and classifying images.
- **Pre-training and Fine-tuning**: The model has been pre-trained on vast image datasets and fine-tuned specifically for image captioning tasks. This ensures that it can capture even subtle details in images, providing highly accurate captions.

### Performance
The Phi 3.5 Vision Instruct model is optimized for real-time captioning, making it suitable for applications like:
- Automatic image annotation
- Content creation for media
- Dataset labeling for machine learning

Its underlying architecture uses deep convolutional neural networks (CNNs) combined with attention-based mechanisms to understand context from the image and generate text that reflects the content accurately.

## Requirements

Make sure your system meets the following requirements:
- Python 3.11 or higher
- 8 GB RAM (recommended for handling image processing tasks)
- An NVIDIA GPU with CUDA support (optional, but recommended for faster processing)
- Libraries: `transformers`, `torch`, `Pillow`, `requests`

If you are using a CPU-only system, the performance may be slower, but the model will still work.

## Acknowledgements

This repository uses the **Phi 3.5 Vision Instruct** model developed by Microsoft. Special thanks to the open-source community and the Hugging Face platform for providing access to powerful AI models.

For more details about the Phi model series, visit [Hugging Face's Phi Model page](https://huggingface.co/microsoft).
