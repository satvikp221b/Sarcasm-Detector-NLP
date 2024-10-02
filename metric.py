import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

json_files = ['logistic_regression_metrics.json', 'lstm_metrics.json', 'bert_metrics.json', 'logistic_regression_feature_metrics.json']

def load_metrics(file_path):
    print(file_path)
    with open(f'metrics/{file_path}', 'r') as f:
        return json.load(f)

metrics_data = {}
for file in json_files:
    model_name = file.split('.')[0].capitalize()
    metrics_data[model_name] = load_metrics(file)

metrics_list = []
for model, metrics in metrics_data.items():
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']
    confusion_matrix = metrics['confusion_matrix']
    
    metrics_list.append([model, accuracy, precision, recall, f1_score, confusion_matrix])

df = pd.DataFrame(metrics_list, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'])


print(df)

# Plotting Confusion Matrices for each model
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  

for i, (model, metrics) in enumerate(metrics_data.items()):
    ax = axes[i // 2, i % 2]  
    cm = metrics['confusion_matrix']
    ax.matshow(cm, cmap='Blues', alpha=0.75)
    for (x, y), value in np.ndenumerate(cm):
        ax.text(y, x, f'{value}', ha='center', va='center')

    ax.set_title(f'{model} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()
