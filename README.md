🩺 Oral Cancer Detection using Machine Learning (CNN & Image Processing)
This project focuses on building an AI-powered system for early detection of Oral Cancer using Deep Learning and Computer Vision techniques. The model classifies oral cavity images into Benign (Non-Cancerous) or Malignant (Cancerous) categories and provides a confidence score. The system also integrates image preprocessing, lesion segmentation, and a user-friendly interface for real-time prediction.

📷 Screenshots
 EXAMPLE 1
1. Upload Image page
   <img width="550" height="402" alt="Picture1" src="https://github.com/user-attachments/assets/f37bc8e4-91d9-4905-9328-6b79e22e5763" />
   
2.Segmented Mask

<img width="444" height="255" alt="Picture4" src="https://github.com/user-attachments/assets/24e793ea-a03c-472d-a5eb-8446a17b3a68" />

3. Decision Result - Negative
   <img width="690" height="537" alt="Picture6" src="https://github.com/user-attachments/assets/7260033a-c637-4861-a1cf-b733939ff014" />

EXAMPLE 2

1.Upload image page
<img width="550" height="402" alt="Picture1" src="https://github.com/user-attachments/assets/c879df73-06e3-41a8-a274-90f9c3065b53" />


2.Segmented Image

<img width="253" height="256" alt="Picture3" src="https://github.com/user-attachments/assets/8ef98be4-ecb4-4e3b-9411-c6e0d7cab7dd" />

3 Decision Result-Positive
<img width="757" height="719" alt="Picture5" src="https://github.com/user-attachments/assets/7414b602-db1d-4c6e-8330-78493d9d5dc3" />


Features

End-to-End Deep Learning Pipeline
Image acquisition, preprocessing, segmentation, and classification.
Image Preprocessing
Noise removal, contrast enhancement, sharpening.
Lesion Segmentation
Isolates ROI (Region of Interest) using Otsu Thresholding and contours.
Deep Learning Model
CNN-based binary classifier with high accuracy.
GUI/CLI Interface
Upload an image and get predictions with confidence scores.

🛠 Tech Stack

Programming Language: Python (3.10+)
Deep Learning Framework: PyTorch, Torchvision
Image Processing: OpenCV
Data Handling: NumPy, Pandas
Environment: Jupyter Notebook / VS Code
Version Control: Git & GitHub

📂 Project Structure

Oral-Cancer-Detection/
│
├── data/
│   ├── original_images/
│   │   ├── CANCER/
│   │   └── NON-CANCER/
│   ├── segmented_lesions/
│
├── models/
│   └── cancer_classifier.pth
│
├── scripts/
│   ├── train_model.py
│   ├── predict_image.py
│   ├── segment_images.py
│
├── requirements.txt
├── README.md
└── report.pdf

📥 Dataset

The dataset consists of oral cavity images classified into:
CANCER (Malignant)
NON-CANCER (Benign)
Dataset sourced from open-access medical repositories and research datasets.
Images were organized in folders for ImageFolder-based loading in PyTorch.

🔍 Methodology

Input Image Acquisition
Collect oral images from datasets or clinical sources.
Preprocessing
Denoising using fastNlMeansDenoisingColored.
Contrast enhancement using CLAHE.
Sharpening using custom kernel filter.
Resizing and normalization.

Segmentation

Convert to grayscale → Otsu Thresholding → Contour detection.
Extract Region of Interest (ROI).
Save segmented lesions for training.

Feature Extraction

CNN-based automatic feature learning.
Additional handcrafted features:
GLCM (Gray Level Co-occurrence Matrix).
GLRLM (Gray Level Run Length Matrix).

Classification

Machine Learning Models Tested:
SVM, KNN, Naïve Bayes (on handcrafted features).
Deep Learning Model:
CNN with:
    3 Convolutional Layers
    ReLU Activation
    Max Pooling
Dropout for regularization
Fully Connected Layers
Loss: Binary Cross-Entropy
Optimizer: Adam

Output
Prediction: Benign or Malignant
Confidence Score for each class.

📊 Performance Analysis
Machine Learning Models

Classifier	Accuracy	Precision	Recall	F1-Score
SVM	           85%	       83%	      84%	 83.5%
KNN	           80%	       78%	      79%	 78.5%
Naive Bayes	   75%	       72%	      74%	 73%

Deep Learning (CNN)

Metric	Training	Validation
Accuracy	97%	      92%
Precision	93%	      91%
Recall	    94%	      92%
F1-Score	93%	     91.5%
Observation: CNN outperformed all ML models with 92% validation accuracy.

🚀 How to Run

Clone the Repository
git clone https://github.com/yourusername/oral-cancer-detection.git
cd oral-cancer-detection
Install Dependencies
pip install -r requirements.txt
Train the Model
python train_model.py
Predict an Image
python predict_image.py

📌 Future Enhancements

Deploy as Web Application using Flask/Django.
Integrate Transfer Learning with pre-trained models like ResNet.
Add Explainability (Grad-CAM) for medical interpretability.
Support Mobile App for Rural Screening.

📜 License
This project is for academic purposes only. For commercial use, proper dataset licensing and regulatory approvals are required.
