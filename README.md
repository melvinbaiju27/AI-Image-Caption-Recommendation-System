# AI Image Caption Recommendation System

A Flask-based web application that uses AI to generate captions for images and provide alternative caption recommendations.

## Features

- Upload images from your device or provide an image URL
- AI-powered image caption generation
- Multiple alternative caption suggestions
- Responsive, modern user interface
- Copy and save functionality for generated captions

## Project Structure

```
flask-image-caption-system/
├── app.py                  # Main Flask application file
├── static/                 # Static files directory
│   └── uploads/            # Directory for uploaded images
└── templates/              # HTML templates
    └── index.html          # Main application template
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flask-image-caption-system.git
cd flask-image-caption-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install flask torch torchvision transformers pillow requests numpy
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

## Technical Details

### Backend (Python/Flask)
- **Flask**: Web framework for handling requests and serving the application
- **PyTorch**: Deep learning framework for running the image captioning model
- **Transformers**: Hugging Face library for pre-trained models
- **PIL (Pillow)**: Image processing library

### Frontend
- **Bootstrap 5**: Frontend framework for responsive design
- **Font Awesome**: Icon library
- **JavaScript**: Client-side interactivity
- **HTML/CSS**: Structure and styling

### AI Model
The application uses a pre-trained Vision Transformer (ViT) and GPT-2 model from Hugging Face that has been fine-tuned for image captioning tasks.

## Potential Enhancements
- User accounts and saved captions history
- Integration with social media platforms
- More sophisticated caption recommendation algorithms
- Caption style options (formal, casual, poetic, etc.)
- Batch processing for multiple images
- Advanced image editing features

## Website Demo
