# 🩺 Multi Cancer Dataset - 8 Types of Cancer Images

## Overview
This dataset contains images of various cancer types, compiled for research and analysis purposes. It includes **8 main cancer classes** and **26 subclasses**, providing a rich resource for medical image classification and machine learning applications.

---

## 📊 Dataset Details & Structure

### **1. Acute Lymphoblastic Leukemia (ALL)** 🔗 [Source](https://www.kaggle.com/datasets/mehradaria/leukemia)
- **Source**: Compiled from images provided in the dataset by Mehrad Aria on Kaggle.
- **Path**: `/ALL`
- **Description**: Contains 20,000 images representing benign and various stages of leukemia.

| **Path**            | **Subclass** | **Description**              |
|---------------------|--------------|------------------------------|
| **/all_benign**     | Benign       | Non-cancerous, healthy cells |
| **/all_early**      | Early        | Early stages of leukemia     |
| **/all_pre**        | Pre          | Pre-stage abnormal cells     |
| **/all_pro**        | Pro          | Advanced leukemia cells      |

---

### **2. Brain Cancer** 🔗 [Source](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- **Source**: Compiled using images provided in a Figshare dataset.
- **Path**: `/Brain Cancer`
- **Description**: Contains 15,000 images covering 3 main types of brain cancer.

| **Path**            | **Subclass**  | **Description**         |
|---------------------|---------------|-------------------------|
| **/brain_glioma**   | Glioma        | Most common brain tumor |
| **/brain_menin**    | Meningioma    | Tumors affecting brain membranes |
| **/brain_tumor**    | Pituitary Tumor | Tumors affecting the pituitary gland |

---

### **3. Breast Cancer** 🔗 [Source](https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset)
- **Source**: Collected from the Breast Cancer dataset by Anas Elmasry on Kaggle.
- **Path**: `/Breast Cancer`
- **Description**: A total of 10,000 images representing benign and malignant types of breast cancer.

| **Path**            | **Subclass**    | **Description**       |
|---------------------|-----------------|-----------------------|
| **/breast_benign**  | Benign          | Non-cancerous breast tissues |
| **/breast_malignant** | Malignant       | Cancerous breast tissues     |

---

### **4. Cervical Cancer** 🔗 [Source](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
- **Source**: Images obtained from Prahlad Mehandiratta’s dataset on Kaggle.
- **Path**: `/Cervical Cancer`
- **Description**: 25,000 images covering various cervical cell types.

| **Path**           | **Subclass**     | **Description**                   |
|--------------------|------------------|-----------------------------------|
| **/cervix_dyk**    | Dyskeratotic     | Abnormal cell growth              |
| **/cervix_koc**    | Koilocytotic     | Cells showing changes from viral infections (e.g., HPV) |
| **/cervix_mep**    | Metaplastic      | Cells changed from one type to another (precancerous) |
| **/cervix_pab**    | Parabasal        | Immature squamous cells           |
| **/cervix_sfi**    | Superficial-Intermediate | More mature squamous cells |

---

### **5. Kidney Cancer** 🔗 [Source](https://www.kaggle.com/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- **Source**: Compiled from the CT Kidney dataset on Kaggle.
- **Path**: `/Kidney Cancer`
- **Description**: Contains 10,000 images of normal and tumorous kidney tissues.

| **Path**            | **Subclass**     | **Description**     |
|---------------------|------------------|---------------------|
| **/kidney_normal**  | Normal           | Healthy kidney tissues |
| **/kidney_tumor**   | Tumor            | Tumor-affected kidney tissues |

---

### **6. Lung and Colon Cancer** 🔗 [Source](https://www.kaggle.com/datasets/biplobdey/lung-and-colon-cancer)
- **Source**: Compiled from the dataset by Biplob Dey on Kaggle.
- **Path**: `/Lung and Colon Cancer`
- **Description**: 25,000 images covering different tissue types in the lung and colon.

| **Path**           | **Subclass**             | **Description**                      |
|--------------------|--------------------------|--------------------------------------|
| **/colon_aca**     | Colon Adenocarcinoma     | Cancerous cells of the colon         |
| **/colon_bnt**     | Colon Benign Tissue      | Healthy colon tissues                |
| **/lung_aca**      | Lung Adenocarcinoma      | Cancerous cells of the lung          |
| **/lung_bnt**      | Lung Benign Tissue       | Healthy lung tissues                 |
| **/lung_scc**      | Lung Squamous Cell Carcinoma | Aggressive lung cancer type      |

---

### **7. Lymphoma** 🔗 [Source](https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification)
- **Source**: Sourced from the Malignant Lymphoma Classification dataset by Andrew MvD on Kaggle.
- **Path**: `/Lymphoma`
- **Description**: Contains 15,000 images across 3 subclasses of lymphoma.

| **Path**            | **Subclass**                     | **Description**                   |
|---------------------|----------------------------------|-----------------------------------|
| **/lymph_cll**      | Chronic Lymphocytic Leukemia     | Slow-progressing blood cancer     |
| **/lymph_fl**       | Follicular Lymphoma              | Slow-growing non-Hodgkin lymphoma |
| **/lymph_mcl**      | Mantle Cell Lymphoma             | Aggressive form of lymphoma       |

---

### **8. Oral Cancer** 🔗 [Source](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset)
- **Source**: Images obtained from Ashenafi Fasil Kebede's dataset on Kaggle.
- **Path**: `/Oral Cancer`
- **Description**: 10,000 images of normal and cancerous oral tissues.

| **Path**           | **Subclass**     | **Description**                      |
|--------------------|------------------|--------------------------------------|
| **/oral_normal**   | Normal           | Healthy oral tissues                 |
| **/oral_scc**      | Oral Squamous Cell Carcinoma | Cancerous oral cells |

---

## 🔄 Data Augmentation & Preprocessing

### Image Augmentation
The dataset was augmented using Keras's `ImageDataGenerator` to enhance diversity and robustness. Below are the parameters used for augmentation:

```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.2, 1.2]
)
```

The augmentations include:

	•	Rotation: Up to 10 degrees.
	•	Width & Height Shift: Up to 10% of the total image size.
	•	Shearing & Zooming: 10% variation.
	•	Horizontal Flip: Randomly flips images for additional diversity.
	•	Brightness Adjustment: Ranges from 0.2 to 1.2 for varying light conditions.

### Image Cropping & Renaming

	•	All images were resized to a consistent 512x512 pixels.
	•	Each file was renamed for uniformity in the format: <subclass>_<serial_number>.jpg.

## 📝 Citation

If you use this dataset in your research or project, please make sure to cite it appropriately. Below is the APA citation format:

**APA**
Obuli Sai Naren. (2022). Multi Cancer Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3415848

For more citation formats, please refer to the `DOI Citation` section on Kaggle.

Thank you for properly attributing the work and contributing to the scientific community! ❤️