# CancerCare

CancerCare is an AI-powered web application designed to detect cancer at early stages using medical images, specifically CT scans. This project integrates genomic data exploration and personalized health information to assist users in understanding their conditions and seeking appropriate care.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Early Cancer Detection:** Uses machine learning models to analyze CT scan images for cancer detection.
- **Genomic Information:** Incorporates genomic sequencing data to identify mutations and biomarkers associated with specific cancer types.
- **Symptom Tracker:** Allows users to log symptoms and receive personalized advice and resources.
- **Multilingual Support:** Offers localized and multilingual cancer-related information and resources.
- **User-Friendly Interface:** Designed with an intuitive interface for easy navigation and interaction.

## Technologies Used
- **Frontend:** HTML, CSS, JavaScript, Next.js
- **Backend:** Flask
- **Machine Learning:** PyTorch, TensorFlow, CNN for image classification
- **APIs:** Ensemble API for genomic information, Gemini API for chatbot support
- **Content Management:** Contentstack for dynamic content delivery
- **Hosting:** [Specify your hosting platform, e.g., AWS, Heroku]

## Model Details
- **Accuracy:** 99.2%
- **Dataset:** Trained on a dataset with more than 100,000 images.
- **Cancer Types:** Can detect over 11 types of cancer at different stages.
- **Metrics:** Evaluated using Mean AUC, AUC-ROC, Specificity, Sensitivity, F1 score, and others.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cancercare.git

Navigate to the project directory:
bash
Copy code
cd cancercare
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
Run the application:
bash
Copy code
python app.py
Open your web browser and go to http://localhost:5000 to access the application.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any changes you'd like to propose.

License
This project is licensed under the MIT License - see the LICENSE file for details.


Thank you for checking out CancerCare! Your support is appreciated in the fight against cancer.

Let me know if you need any further modifications!
