from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from DataProcessing import X_train_lr, Y_train_lr, X_test_lr, Y_test_lr
import matplotlib.pyplot as plt


logReg = LogisticRegression(max_iter=1000, random_state=42)
logReg.fit(X_train_lr, Y_train_lr)

y_pred=logReg.predict(X_test_lr)

# Accuracy
accuracy = accuracy_score(Y_test_lr, y_pred)

# Precision
precision = precision_score(Y_test_lr, y_pred, average='weighted')

# Recall
recall = recall_score(Y_test_lr, y_pred, average='weighted')

# F1 Score
f1 = f1_score(Y_test_lr, y_pred, average='weighted')

# Confusion Matrix
cm = confusion_matrix(Y_test_lr, y_pred)

print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')
print('Confusion Matrix:')
print(cm)
