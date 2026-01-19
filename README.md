# Detecting-Crime-Related-Content

Project Overview:
This capstone project focuses on the **automatic detection of crime-related content** from images using **Machine Learning (ML) and Deep Learning (DL)** techniques.  

With the rapid growth of digital media and social platforms, violent and illegal content such as **weapon imagery and crime-related visuals** has become increasingly prevalent. Manual moderation is inefficient and error-prone, creating a need for **automated, scalable detection systems**.

This project implements and evaluates multiple **deep learning architectures** using transfer learning to identify crime-related visual content accurately.

Models Implemented:
The following deep learning models were implemented and evaluated:

- **VGG16**
- **VGG19**
- **InceptionNet V3**
- **MobileNet**

Technologies Used:
- **Programming Language:** Python  
- **Deep Learning Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn  
- **Development Environment:** Google Colab / Jupyter Notebook

Dataset Description:
- **Dataset Used:** OD-Weapon Detection Dataset (sourced from GitHub)
- **Data Type:** Image data
- **Classes:** Crime-related vs Non-crime-related content

Methodology:

1️⃣ Data Preprocessing
- Image normalization
- Reshaping images for CNN input
- Dataset splitting into training and validation sets

2️⃣ Model Architecture
- Pretrained base models (ImageNet weights)
- Custom classification layers:
  - Flatten layer
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Softmax output layer

3️⃣ Model Training
- Transfer learning applied (base model weights frozen)
- Optimization using Adam optimizer
- Categorical cross-entropy loss function

4️⃣ Model Evaluation
- Accuracy and loss analysis
- Confusion matrices
- Classification reports (Precision, Recall, F1-Score)

5️⃣ Performance Enhancement
- Fine-tuning selected layers
- Dimensionality reduction using **PCA**
- Clustering analysis using **K-Means**

MobileNet demonstrated the best balance between **accuracy and computational efficiency**.
