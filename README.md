# Emotion & Gender Detection App

This is a Streamlit app that detects faces, emotions, and gender in images using deep learning models.

## Features

- Face detection using Haar Cascade classifier
- Emotion detection (angry, sad, happy, surprise, neutral)
- Gender detection (male/female)
- Support for both image upload and webcam input
- Real-time processing with visual feedback

## How to Use

1. **Image Upload**:
   - Click on the "Choose an image" button to upload an image
   - Click "Process Uploaded Image" to analyze the image
   - The processed image will show detected faces with emotion and gender labels

2. **Webcam**:
   - Click "Start Webcam" to activate your camera
   - Take a picture using the webcam interface
   - The app will automatically process the image and show the results

## Technical Details

The app uses three pre-trained models:
- Haar Cascade classifier for face detection
- Mini Xception model for emotion detection (trained on FER2013 dataset)
- Mini Xception model for gender detection (trained on IMDB dataset)

## Local Development

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## License

This project is open source and available under the MIT License.
