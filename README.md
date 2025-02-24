# Customer Segmentation using K-Means Clustering

This project is a **Streamlit-based web application** for customer segmentation using **K-Means Clustering**. Users can upload their dataset, select features, and perform clustering to understand different customer segments based on spending behavior and income levels.

## 🚀 Features
- **Upload CSV Dataset**: Supports customer data files.
- **Feature Selection**: Choose numerical features for clustering.
- **Elbow Method & Silhouette Score**: Helps determine the optimal number of clusters.
- **Data Preprocessing**: Standardizes data using `StandardScaler`.
- **K-Means Clustering**: Dynamically assigns customers into clusters.
- **Cluster Interpretation**: Labels clusters based on spending and income levels.
- **Visualization**: Scatter plot with cluster centroids.

## 🛠 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run app.py
```

## 📂 Project Structure
```
📂 customer-segmentation-kmeans
├── app.py             # Streamlit application
├── requirements.txt   # Required Python packages
├── README.md          # Project documentation
└── sample_data.csv    # Example dataset (if available)
```

## 📊 Usage
1. Upload a customer dataset (`CSV` format).
2. Select numerical features for clustering.
3. Choose the number of clusters (K) or let the app suggest the best K.
4. View clustered data, segment interpretation, and visualizations.

## 📜 Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 🎯 Example Dataset
Ensure your dataset contains numerical features like:
```
CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
1, Male, 19, 15, 39
2, Male, 21, 15, 81
3, Female, 20, 16, 6
...
```

## 📢 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## 📄 License
This project is open-source and available under the MIT License.

---
Made with ❤️ using Streamlit & Scikit-learn

