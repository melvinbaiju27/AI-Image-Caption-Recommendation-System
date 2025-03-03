# app.py
from flask import Flask, render_template, request, jsonify, url_for
import torch
from PIL import Image
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import requests
from io import BytesIO
import os
import uuid
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ImageCaptionSystem:
    def __init__(self):
        # Load pre-trained model
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set generation parameters
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
        
        # Sample alternative captions
        self.alternative_captions = [
            "people enjoying a beautiful sunset at the beach",
            "a group of friends having dinner together",
            "a person hiking through a forest trail",
            "a dog playing in a park with its owner",
            "a busy city street with people shopping",
            "a peaceful mountain landscape with snow",
            "children playing in a playground",
            "a family gathering for a celebration",
            "an athlete competing in a sporting event",
            "a cozy cafe with people reading books",
            "a couple walking hand in hand in a park",
            "a picturesque village with colorful houses",
            "a boat sailing on a calm lake",
            "a cat lounging on a windowsill",
            "a vibrant farmers market with fresh produce",
            "a breathtaking view from a mountain peak",
            "a bustling food street with vendors",
            "a serene river flowing through a valley",
            "a beach with crystal-clear water",
            "a forest with tall, ancient trees",
            "a group of people camping under the stars",
            "a person riding a bicycle on a scenic trail",
            "a traditional festival with colorful decorations",
            "a quiet library with shelves of books",
            "a city skyline at dusk",
            "a couple dancing at a wedding",
            "a snowy landscape with a cozy cabin",
            "a group of people playing a board game",
            "a waterfall cascading into a pool",
            "a sunflower field in full bloom",
            "a hot air balloon floating in the sky",
            "a person painting on an easel",
            "a scenic train ride through the countryside",
            "a majestic castle on a hill",
            "a family having a picnic in a park",
            "a person kayaking on a river",
            "a lively street parade with performers",
            "a cozy fireplace with a cup of hot cocoa",
            "a person doing yoga on a beach",
            "a colorful coral reef underwater",
            "a child blowing bubbles in the yard",
            "a couple watching a movie at home",
            "a person fishing by a lake",
            "a serene garden with blooming flowers",
            "a night sky full of stars",
            "a charming café with outdoor seating",
            "a historic landmark lit up at night",
            "a person playing guitar by a campfire",
            "a group of people on a road trip",
            "a surfer catching a wave",
            "a chef cooking in a busy kitchen",
            "a picturesque vineyard with grapevines",
            "a person meditating by the ocean",
            "a bustling market with colorful stalls",
            "a person reading a book in a hammock",
            "a couple sharing a romantic dinner",
            "a group of kids flying kites",
            "a person practicing martial arts",
            "a beach bonfire with friends",
            "a quaint village with cobblestone streets",
            "a person rock climbing on a cliff",
            "a family building a sandcastle",
            "a person playing the piano",
            "a beautiful garden with a fountain",
            "a person snorkeling in clear water",
            "a group of friends on a camping trip",
            "a person running through a park",
            "a cozy cabin in the woods",
            "a person playing soccer on a field",
            "a couple watching the sunrise",
            "a person photographing wildlife",
            "a colorful autumn forest",
            "a group of people doing a workout",
            "a person riding a horse",
            "a scenic drive along the coast",
            "a person baking in the kitchen",
            "a couple enjoying a picnic by a lake",
            "a person swimming in a pool",
            "a bustling harbor with boats",
            "a person playing with a cat",
            "a beautiful garden with butterflies",
            "a person skateboarding in a park",
            "a family celebrating a birthday",
            "a serene pond with lily pads",
            "a person practicing yoga in a park",
            "a couple exploring a historic city",
            "a group of friends playing beach volleyball",
            "a person bird-watching in a forest",
            "a cozy bookstore with shelves of books",
            "a person exploring a cave",
            "a beautiful rainbow over a field",
            "a couple having a coffee date",
            "a person painting a mural",
            "a serene beach with palm trees",
            "a person hiking a mountain trail",
            "a group of people at a music festival",
            "a person sailing on a boat",
            "a vibrant cityscape at night",
            "a person gardening in their backyard",
            "a couple stargazing on a clear night",
            "a person doing pottery in a studio",
            "a cozy living room with a fireplace",
            "a person jogging along a river",
            "a family at an amusement park",
            "a person swimming with dolphins",
            "a beautiful sunset over a lake",
            "a person playing chess in a park",
            "a vibrant carnival with rides and games",
            "a person snowboarding down a mountain",
            "a person enjoying a spa day",
            "a serene meadow with wildflowers",
            "a person playing a video game",
            "a beautiful sunrise over a beach",
            "a person practicing archery",
            "a group of friends on a boat trip",
            "a person playing with a puppy",
            "a beautiful flower garden",
            "a person doing a puzzle at home",
            "a couple enjoying a scenic hike",
            "a person taking a road trip",
            "a serene lake surrounded by mountains",
            "a person practicing ballet",
            "a group of friends having a barbecue",
            "a person reading in a cozy nook",
            "a couple enjoying a beach sunset",
            "a person exploring a botanical garden",
            "a vibrant street fair with food vendors",
            "a person riding a scooter in a city",
            "a group of friends at a karaoke night",
            "a person visiting an art gallery",
            "a couple enjoying a hot air balloon ride",
            "a person doing tai chi in a park",
            "a group of people at a science museum",
            "a person playing tennis on a court",
            "a serene waterfall in a tropical forest",
            "a person practicing meditation",
            "a couple walking on a quiet beach",
            "a person exploring a historic castle",
            "a group of people at a wine tasting",
            "a person enjoying a sunset cruise",
            "a couple taking a cooking class",
            "a person exploring a desert landscape",
            "a group of friends playing basketball",
            "a person relaxing in a hammock by the sea",
            "a couple having a romantic dinner on a terrace",
            "a person paddleboarding on a calm lake"
        ]

    
    def predict_caption(self, image):
        """Generate a caption for the provided image"""
        if image is None:
            return ""
            
        images = []
        images.append(image)
        
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        return preds[0].strip()
    
    def get_recommendations(self, base_caption, num_recommendations=3):
        """Generate caption recommendations based on the initial caption"""
        # In a real system, this would use NLP techniques to generate variations
        recommendations = []
        selected_alts = random.sample(self.alternative_captions, min(num_recommendations+2, len(self.alternative_captions)))
        
        # Create some variation in the captions
        for alt in selected_alts[:num_recommendations]:
            if random.random() > 0.5:
                recommendations.append(f"{base_caption} with {alt.split(' ', 1)[1]}")
            else:
                recommendations.append(f"{alt.capitalize()} similar to {base_caption}")
                
        return recommendations

# Initialize the caption system
caption_system = ImageCaptionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files and 'image_url' not in request.form:
        return jsonify({'error': 'No file or URL provided'}), 400
    
    try:
        image = None
        
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert("RGB")
            image_path = url_for('static', filename=f'uploads/{filename}')
        
        # Handle URL input
        elif 'image_url' in request.form and request.form['image_url']:
            url = request.form['image_url']
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Save the image from URL
            filename = str(uuid.uuid4()) + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            image_path = url_for('static', filename=f'uploads/{filename}')
        
        # Generate caption and recommendations
        num_recommendations = int(request.form.get('num_recommendations', 3))
        caption = caption_system.predict_caption(image)
        recommendations = caption_system.get_recommendations(caption, num_recommendations)
        
        return jsonify({
            'success': True,
            'image_path': image_path,
            'caption': caption,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)