# **Unemployment Rate Prediction**

A data science project analyzing and predicting unemployment rates in India using historical data. This project includes data visualization, statistical analysis, and a simple predictive model implemented in Python. A REST API is also created for prediction using the trained model.

---

## **Features**
- Analyze unemployment trends over time and by region.
- Visualize data using histograms, line plots, and boxplots.
- Train a linear regression model to predict unemployment rates.
- Deploy the trained model via a REST API built with Flask.

---

## **Dataset**
The dataset contains unemployment statistics across India, including:
- `Date`: The date of the data point.
- `Region`: The region associated with the data point.
- `Estimated Unemployment Rate (%)`: The unemployment rate.

**Source**: [url = "https://storage.googleapis.com/kagglesdsdata/datasets/752131/1621146/Unemployment%20in%20India.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241002%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241002T124609Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4025cdb18a43b4a2412d8ca67f80ed8e29f1d0d458d1360fd6e1eee632e33b16bff0d4ed962bcfb03d0d77c036c7b64f54dad34f30ba4842157282121e38497732f94cb217724269a17606d77f703ba9ee8623bff6a9802f2bfd979db32f03234775efa1b0af501512d4db2c55053ddc7b06dc93065d44fcfc48f25e98a932129fb314692ae6adf9e7405fff5902e52f468c3c23e78119e98ee5c7b996a77c1b5c73acf76aba61388b0c4209234d8c3a205b7ca7398c89f979c94b09182605ee34db0701ebf87842d597444af2089420a583c0912e1c11960421cc56090ac6ee41e122c61c2b1e33e528b52edf98e8f7c355b12c132178e8748adb40e6bf18e7"
] 
---

## **Project Structure**
```
unemployment-rate-prediction/
├── unemployment_rate_notebook.ipynb  # Jupyter Notebook containing all the code
├── unemployment_rate.pkl             # Trained model file
├── requirements.txt                  # Dependencies for the project
├── app.py                            # Flask API script
├── README.md                         # Project documentation
└── outputs/                          # Saved plots and outputs 
```

---

## **Dependencies**
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `flask`
- `joblib`

Install them using:
```bash
pip install -r requirements.txt
```

---

## **How to Run**

### **1. Interactive Mode**
1. Open the `unemployment_rate_notebook.ipynb` in Jupyter Notebook or VSCode.
2. Run all cells sequentially to:
   - Load and preprocess the dataset.
   - Visualize trends.
   - Train a predictive model.
   - Test the Flask API locally.
---

## **Results**
### **Visualizations**
- **Distribution of Unemployment Rate**: Shows the spread of unemployment percentages.
- **Trends Over Time**: Line chart tracking unemployment rates over months and years.
- **Unemployment by Region**: Boxplot comparing unemployment rates across regions.

### **Model Performance**
- **Mean Squared Error (MSE)**: Quantifies the model's predictive accuracy. 
  ```plaintext
  Mean Squared Error: 126.99221626406772

  ---

**Notebook Sharing**:
   - [Binder]["https://mybinder.org/v2/gh/Aliyasuv/OIBSIP/main"] 

## **License**
All rights reserved.

---

## **Contact**
- **GitHub**: [Aliyasuv](https://github.com/Aliyasuv)
- **Email**: aliya.ansari1685@gmail.com
