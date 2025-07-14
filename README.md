# Titanic Survival Prediction using Deep Learning

This project tackles the well-known Kaggle competition "Titanic - Machine Learning from Disaster" using deep learning techniques. The goal is to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, and family relationships.

While many traditional approaches rely on tree-based models, this project leverages a neural network model built with TensorFlow/Keras, incorporating data preprocessing, feature engineering, and model evaluation.

---

## Project Overview

- Problem Type: Binary classification (Survived = 0 or 1)  
- Approach: Deep neural network (DNN)  
- Libraries: TensorFlow, Keras, Pandas, NumPy, Scikit-learn

---

## Dataset

The dataset is provided by Kaggle and includes two CSV files:

- `train.csv`: Contains labels (Survived) and features  
- `test.csv`: Contains only features; used for final prediction

Key features used:

- Pclass: Ticket class (1st, 2nd, 3rd)  
- Sex: Gender of the passenger  
- Age: Age in years  
- SibSp: Number of siblings/spouses aboard  
- Parch: Number of parents/children aboard  
- Fare: Ticket fare  
- Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Data Preprocessing

Before training, the following steps were applied:

- Missing value imputation:  
  - Age and Fare were filled using median values.  
  - Embarked was filled using the most frequent value (mode).  
- Categorical encoding:  
  - Sex and Embarked columns were converted to numerical form.  
- Feature scaling:  
  - All numeric features were standardized using StandardScaler.

---

## Model Architecture

A deep feedforward neural network was built using TensorFlow and Keras. The architecture includes:

- Input layer matching the number of features  
- Hidden layers with ReLU activation functions  
- Final output layer with sigmoid activation for binary classification  
- Loss function: Binary Crossentropy  
- Optimizer: Adam  
- Evaluation metric: Accuracy

---

## Training

The model was trained using the training dataset with an 80/20 train-validation split. Key training details:

- Epochs: 100  
- Batch size: 32  
- Early stopping and dropout can be added to reduce overfitting (optional)

---

## Evaluation

After training, the model was evaluated on unseen test data. Model performance was measured using:

- Accuracy score  
- Confusion matrix  
- Classification report (precision, recall, F1-score)

---

## Conclusion

This project demonstrates the use of deep learning for a classic binary classification problem. While deep learning can perform well, results depend heavily on:

- Data quality  
- Feature engineering  
- Network tuning

It is also important to compare neural networks with simpler models like logistic regression or random forests, especially when the dataset is relatively small and structured, like Titanic.
