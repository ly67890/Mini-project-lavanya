import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Create a synthetic loan dataset
np.random.seed(42)
data = {
    'ApplicantIncome': np.random.randint(1000, 10000, 200),  # Applicant income
    'CoapplicantIncome': np.random.randint(0, 5000, 200),  # Co-applicant income
    'LoanAmount': np.random.randint(100, 600, 200),  # Loan amount
    'LoanTerm': np.random.choice([15, 30], 200),  # Loan term in years
    'CreditScore': np.random.randint(300, 850, 200),  # Credit score
    'LoanApproved': np.random.choice([0, 1], 200)  # Loan approval (0 = No, 1 = Yes)
}

df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Exploratory Data Analysis
sns.countplot(x='LoanApproved', data=df)
plt.title('Loan Approval Distribution')
plt.show()

# Define features and target variable
X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanTerm', 'CreditScore']]
y = df['LoanApproved']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
