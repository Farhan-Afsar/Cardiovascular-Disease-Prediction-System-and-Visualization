from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from io import BytesIO
import base64

dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/SP5/SP5 Project/Cardiovascular-Disease-Prediction-System-main/Cardiovascular-Disease-Prediction-System-main/SP5/home/templates/cardio_train.csv',sep = ";")

def home(request):
    return render(request, 'home.html')

def first(request):
    return render(request,'first.html')

def about(request):
    return render(request,'about.html')

def random(request):
    return render(request,'random.html')

def logistic(request):
    return render(request,'logistic.html')

def knn(request):
    return render(request,'knn.html')

def tree(request):
    return render(request,'tree.html')


def visual(request):
    return render(request,'visual.html')

def visual2(request):
    return render(request,'visual2.html')

def visual3(request):
    return render(request,'visual3.html')

def visual4(request):
    return render(request,'visual4.html')

def accuracy_metrics(request):
    return render(request,'accuracy.html')

def result(request):
    dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/SP5/SP5 Project/Cardiovascular-Disease-Prediction-System-main/Cardiovascular-Disease-Prediction-System-main/SP5/home/templates/cardio_train.csv',sep = ";")

    dataset.drop('id', axis=1, inplace=True)
    dataset['age'] = (dataset['age'] / 365).astype('int')
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=1)
    KN = KNeighborsClassifier()
    KN.fit(xtrain, ytrain)

    input1 = int(request.GET.get('age'))
    input2 = int(request.GET.get('gender'))
    input3 = int(request.GET.get('height'))
    input4 = float(request.GET.get('weight'))
    input5 = int(request.GET.get('ap_hi'))
    input6 = int(request.GET.get('ap_lo'))
    input7 = int(request.GET.get('cholesterol'))
    input8 = int(request.GET.get('glucose'))
    input9 = int(request.GET.get('smoke'))
    input10 = int(request.GET.get('alco'))
    input11 = int(request.GET.get('active'))

    pred = KN.predict([[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11]])

    result1 = " "

    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request,'knn.html',{"result2": result1})



def visual(request):
    # Load dataset
    dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/untitled folder/SP5/home/templates/cardio_train.csv', sep=";")
    
    # Feature names
    features = {
        'age': 'Age (days)',
        'height': 'Height (cm)',
        'weight': 'Weight (kg)',
        'gender': 'Gender',
        'ap_hi': 'Systolic blood pressure',
        'ap_lo': 'Diastolic blood pressure',
        'cholesterol': 'Cholesterol',
        'gluc': 'Glucose',
        'smoke': 'Smoking',
        'alco': 'Alcohol intake',
        'active': 'Physical activity',
        'cardio': 'Cardiovascular disease'
    }

    # Rename columns
    dataset.rename(columns=features, inplace=True)

    # Check for null values
    null_check = dataset.isnull().sum()

    # Plot NULL Count Analysis
    plt.figure(figsize=(10, 6))
    sns.barplot(x=null_check.index, y=null_check.values)
    plt.xticks(rotation=45)
    plt.title('NULL Count Analysis')
    plt.xlabel('Features')
    plt.ylabel('Count of NULL Values')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_nullcheck = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Plot Outcome Count
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Cardiovascular disease', data=dataset)
    plt.title('Outcome Count')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_outcome = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Generate pairplot
    plt.figure(figsize=(10, 6))
    sns.pairplot(dataset.sample(500))
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_pairplot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Additional visualizations
    # 1. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='Age (days)', data=dataset)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_boxplot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 2. Violinplot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Cardiovascular disease', y='Systolic blood pressure', data=dataset)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_violinplot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Add more visualizations as needed...

    # Render the HTML template with the data
    return render(request, 'visual.html', {
        'plot_html_before': plot_html_nullcheck,
        'plot_html_after': plot_html_outcome,
        'plot_html_pairplot': plot_html_pairplot,
        'plot_html_boxplot': plot_html_boxplot,
        'plot_html_violinplot': plot_html_violinplot,
        'accuracy': list(features.values()),
        'nullcheck': null_check.items()
    })

def visual2(request):
    # Load dataset
    dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/untitled folder/SP5/home/templates/cardio_train.csv', sep=";")
    
    # Additional visualizations
    # 3. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x='age', kde=True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_histogram = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 4. Scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='weight', y='height', data=dataset, hue='ap_hi')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_scatterplot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 5. Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_heatmap = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 6. Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='cholesterol', y='ap_hi', hue='gender', data=dataset)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_barplot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 7. Pie Chart
    plt.figure(figsize=(8, 8))
    dataset['gender'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.ylabel('')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_piechart = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Render the HTML template with the data
    return render(request, 'visual2.html', {
        'plot_html_histogram': plot_html_histogram,
        'plot_html_scatterplot': plot_html_scatterplot,
        'plot_html_heatmap': plot_html_heatmap,
        'plot_html_barplot': plot_html_barplot,
        'plot_html_piechart': plot_html_piechart,
    })


def visual3(request):
    # Load dataset
    dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/untitled folder/SP5/home/templates/cardio_train.csv', sep=";")
    


    # 2. Scatterplot for Blood Pressure
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ap_hi', y='ap_lo', data=dataset, hue='cardio')
    plt.title('Scatterplot of Blood Pressure')
    plt.xlabel('ap_hi')
    plt.ylabel('ap_lo')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_scatterplot_bp = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 3. Boxplot for Cholesterol and Gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cholesterol', y='gender', data=dataset)
    plt.title('Boxplot of Cholesterol by Gender')
    plt.xlabel('cholesterol')
    plt.ylabel('gender')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_boxplot_chol_gender = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 4. Pie Chart for Gender Distribution
    plt.figure(figsize=(8, 8))
    dataset['gender'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.ylabel('')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_piechart_gender = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 5. Bar Plot for Alcohol Intake and Smoking
    plt.figure(figsize=(10, 6))
    sns.countplot(x='alco', hue='smoke', data=dataset)
    plt.title('Bar Plot of Alcohol Intake and Smoking')
    plt.xlabel('alco')
    plt.ylabel('smoke')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_barplot_alco_smoke = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Render the HTML template with the data
    return render(request, 'visual3.html', {
        'plot_html_scatterplot_bp': plot_html_scatterplot_bp,
        'plot_html_boxplot_chol_gender': plot_html_boxplot_chol_gender,
        'plot_html_piechart_gender': plot_html_piechart_gender,
        'plot_html_barplot_alco_smoke': plot_html_barplot_alco_smoke,
    })
def visual4(request):
    # Load dataset
    dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/untitled folder/SP5/home/templates/cardio_train.csv', sep=";")
    
    # Additional visualizations
    # 1. Histogram for Weight (kg)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x='weight', kde=True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_weight_histogram = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 2. Boxplot for Age (days) and Gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='gender', y='age', data=dataset)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_age_boxplot_gender = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 3. Countplot for Glucose levels and Cholesterol
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gluc', hue='cholesterol', data=dataset)
    plt.title('Count Plot of Glucose and Cholesterol')
    plt.xlabel('Glucose')
    plt.ylabel('Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_gluc_cholesterol_countplot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 4. Violinplot for Blood Pressure and Gender
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='gender', y='ap_hi', data=dataset)
    plt.title('Violin Plot of Systolic Blood Pressure by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Systolic Blood Pressure')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_html_bp_violinplot_gender = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Render the HTML template with the data
    return render(request, 'visual4.html', {
        'plot_html_weight_histogram': plot_html_weight_histogram,
        'plot_html_age_boxplot_gender': plot_html_age_boxplot_gender,
        'plot_html_gluc_cholesterol_countplot': plot_html_gluc_cholesterol_countplot,
        'plot_html_bp_violinplot_gender': plot_html_bp_violinplot_gender,
    })

def accuracy_metrics(request):
    # Sample data (replace this with your actual data)
    algorithms = ['KNN', 'Random Forest', 'Decision Tree', 'Logistic Regression']
    accuracy = [68, 70, 63, 70]
    recall = [66, 69, 61, 66]
    precision = [70, 72, 64, 73]
    f1_score = [68, 70, 63, 69]

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Algorithm': algorithms,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1_score
    })

    # Create visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Accuracy', data=metrics_df)
    plt.title('Accuracy Comparison')
    plt.xlabel('Algorithm')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    accuracy_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Recall', data=metrics_df)
    plt.title('Recall Comparison')
    plt.xlabel('Algorithm')
    plt.ylim(0, 100)
    plt.ylabel('Recall')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    recall_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Precision', data=metrics_df)
    plt.title('Precision Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Precision')
    plt.ylim(0, 100)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    precision_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='F1 Score', data=metrics_df)
    plt.title('F1 Score Comparison')
    plt.xlabel('Algorithm')
    plt.ylim(0, 100)
    plt.ylabel('F1 Score')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    f1_score_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Render the HTML template with the data
    return render(request, 'accuracy.html', {
        'accuracy_plot': accuracy_plot,
        'recall_plot': recall_plot,
        'precision_plot': precision_plot,
        'f1_score_plot': f1_score_plot,
    })