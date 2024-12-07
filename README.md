# BiT-ImageCaptioning

**BiT-ImageCaptioning** is a Python package for generating **Arabic image captions**. It was desined as a part of the [Modualr Arabic Visual Question Answering System (ArabicVQA)](https://github.com/shahadMAlshalawi/ArabicVQA.git). This library is designed to provide Arabic captions for images by leveraging the pre-trained **Bidirectional Transformers (BiT)** presented in the paper [Arabic Image Captioning using Pre-training of Deep Bidirectional Transformers](https://aclanthology.org/2022.inlg-main.4/).


## Installation

Clone the repository and install the package locally:

```bash
git clone https://github.com/shahadMAlshalawi/BiT-ImageCaptioning
cd BiT-ImageCaptioning
pip install -e .
```

If you're working in a Jupyter Notebook, restart the environment after installation:

```python
import os
os.kill(os.getpid(), 9)
```



## Quick Start

```python
from BiTImageCaptioning.generation import generate_caption
from BiTImageCaptioning.feature_extraction import ImageFeatureExtractor

# Extract image features
extractor = ImageFeatureExtractor()
image_features = extractor.extract_features("path_to_image.jpg")

# Generate a caption
caption = generate_caption(model="path_to_model", image_features=image_features)
print("Generated Caption:", caption)
```



## Package Architecture

The package is modularly designed to make it easy to understand, extend, and use. Below is the file structure of the package:

```
bit_image_captioning/
│
├── src/
│   ├── bit_image_captioning/               # Main package directory
│   │   ├── __init__.py                     # Package initialization
│   │   ├── datasets/                       # Dataset preparation
│   │   │   ├── __init__.py
│   │   │   └── image_captioning.py         # Dataset class for Image Captioning
│   │   │
│   │   ├── feature_extractors/             # Feature extraction
│   │   │   ├── __init__.py
│   │   │   ├── base.py                     # Base feature extractor interface
│   │   │   ├── vinvl.py                    # VinVL feature extractor
│   │   │   └── custom.py                   # Custom feature extractor (if needed)
│   │   │
│   │   ├── modeling/                       # Model implementation
│   │   │   ├── __init__.py
│   │   │   ├── modeling_bert.py            # Core captioning model using AraBERT
│   │   │   └── configuration.py            # Configuration for AraBERT and feature extractors
│   │   │
│   │   ├── pipelines/                      # Hugging Face-style pipelines
│   │   │   ├── __init__.py
│   │   │   └── image_captioning.py         # Pipeline for generating captions
│   │   │
│   │   ├── tokenizers/                     # Tokenization utilities
│   │   │   ├── __init__.py
│   │   │   └── bert_tokenizer.py           # Tokenizer specific to AraBERT
│   │   │
│   │   ├── utils/                          # General utilities
│   │   │   ├── __init__.py
│   │   │   ├── logging.py                  # Logging functionality
│   │   │   ├── data_processing.py          # Preprocessing utilities
│   │   │   ├── visualization.py            # Visualization tools for captions
│   │   │   └── metrics.py                  # Evaluation metrics (BLEU, CIDEr, ROUGE)
│   │   │
│   │   ├── cli/                            # Command-line interface
│   │   │   ├── __init__.py
│   │   │   └── captioning_cli.py           # CLI for generating captions
│   │   │
│   │   └── evaluation/                     # Caption evaluation tools
│   │       ├── __init__.py
│   │       ├── bleu.py                     # BLEU score calculation
│   │       ├── cider.py                    # CIDEr score calculation
│   │       └── rouge.py                    # ROUGE score calculation
│   │
│
├── notebooks/                              # Demonstration notebooks
│   ├── demo_image_captioning.ipynb         # Example for caption generation
│   └── demo_custom_extractor.ipynb         # Example for custom feature extractors
│
├── LICENSE                                 # License file
├── README.md                               # Documentation
├── requirements.txt                        # Package requirements
├── setup.py                                # Package installation script
└── .gitignore                              # Git ignore file                
```



## Core Components

### 1. Feature Extraction
- File: `feature_extraction.py`
- Extracts features from images using pre-trained models.

```python
from BiTImageCaptioning.feature_extraction import ImageFeatureExtractor

extractor = ImageFeatureExtractor()
image_features = extractor.extract_features("path_to_image.jpg")
```

### 2. Caption Generation
- File: `generation.py`
- Generates captions based on extracted image features.

```python
from BiTImageCaptioning.generation import generate_caption

caption = generate_caption(model="path_to_model", image_features=image_features)
print("Generated Caption:", caption)
```

### 3. Configuration and Utilities
- File: `configuration.py`
- Manages model configurations and utility functions.


## Jupyter Notebooks

The `notebooks/` directory contains several Jupyter Notebooks to help you get started:

- `dataset.ipynb`: Demonstrates how to prepare datasets for training and evaluation.
- `evaluation.ipynb`: Shows how to evaluate the model on test data.
- `inference.ipynb`: Guides you through generating captions for your images.




