import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt
import json

def test_model():
    x_test_data = np.load("./x_test.npy", allow_pickle=True)
    y_test_data = np.load("./y_test.npy", allow_pickle=True)

    model = joblib.load("./model.pkl")
    y_pred = model.predict(x_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    accuracy = round(accuracy * 100, 1)
    
    metrics = {"accuracy": accuracy}
    
    with open('metrics.json', 'w') as f:
       json.dump(metrics, f)

    # Confusion Matrix and plot
    cm = confusion_matrix(y_test_data, model.predict(x_test_data))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.tight_layout()
    plt.savefig("cm.png",dpi=120) 
    plt.close()
    # Print classification report
    cr = classification_report(y_test_data, model.predict(x_test_data))
    with open('classification_report.txt', 'w') as f:
        f.write(str(cr))
    # Plot the ROC curve
    model_ROC = plot_roc_curve(model, x_test_data, y_test_data)
    plt.tight_layout()
    plt.savefig("roc.png",dpi=120) 
    plt.close()

    

if __name__ == '__main__':

    print('Testing PMR model on validation set...')
    test_model()
