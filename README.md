# Smart Waste Classification System

AI-powered web app that detects waste in an image, recognizes the surrounding environment, and gives context-aware disposal recommendations.

**Live demo:** *[smart-waste-classification](https://smart-waste-classification-nbjubnmu8dff9zzeq9wly8.streamlit.app/)*

## Features

- Detects waste objects using YOLOv8
- Recognizes environment using a pretrained Places365 model
- Maps scenes to project classes: beach, home, park, street, market, industrial, other
- Gives disposal / recycling recommendations
- Image upload and camera input
- Deployed with Streamlit

## Waste classes

- plastic
- paper
- cardboard
- glass
- metal
- organic

## How it works

1. User uploads an image or takes a photo
2. YOLOv8 detects waste items and draws bounding boxes
3. Places365 predicts the scene
4. Scene labels are mapped to a simple environment class
5. A rule-based engine generates recommendations

Example: plastic detected on a beach → high priority collection to prevent water pollution.

## Tech stack

- Python
- YOLOv8 (Ultralytics)
- Places365 (ResNet18)
- Streamlit
- PyTorch
- Google Colab for training
- Streamlit Community Cloud for deployment

## Datasets

- Waste detection: public Roboflow object detection dataset  
  Classes: paper, plastic, glass, metal, cardboard, organic
- Environment recognition: pretrained Places365 scene model  
  Outputs are mapped to project environment classes

## Project structure

```text
smart-waste-classification/
├── app.py
├── best.pt
├── requirements.txt
├── runtime.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads Places365 weights automatically.

## Deployment
The app is deployed on Streamlit Community Cloud from this GitHub repository.

## Limitations
- Crowded images may miss some waste items
- Plastic is detected more reliably than metal or plain paper
- Environment recognition can confuse a heavily littered beach with a landfill
- No custom Indian street/home dataset was collected

## Future work
- Fine-tune environment recognition on local images
- Improve detection of paper, metal, and small objects
- Add a mobile app
- Integrate with smart bins or civic reporting

## Team

- Jnanesh M
- Harshith Kumar
        
MCA project, Srinivas University

Smart Waste Classification System using YOLOv8 Object Detection and Environment Recognition
