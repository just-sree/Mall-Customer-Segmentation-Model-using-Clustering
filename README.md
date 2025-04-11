# 🛍️ Mall Customer Segmentation & Clustering App

This Streamlit application allows users to segment mall customers based on key behavioral features such as income and spending score. Using unsupervised learning (KMeans Clustering), users can explore patterns in customer behavior and visualize meaningful insights.

---

## 📌 Overview

This project applies machine learning techniques (KMeans) for customer segmentation. It includes:
- Data cleaning and preprocessing
- Clustering using KMeans
- Elbow method and silhouette analysis for optimal clusters
- Interactive visualizations using Streamlit
- Simple, user-driven cluster analysis

---

## 🛠 Project Structure

```
unsupervised_clustering_app/
├── app.py
├── requirements.txt
├── data/
│   └── mall_customers.csv
├── notebooks/
│   └── Unsupervised_Clustering_Solution.ipynb
├── utils/
│   ├── clustering.py
│   └── preprocessing.py
└── README.md
```

---

## 🚀 How to Run

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/unsupervised_clustering_app.git
cd unsupervised_clustering_app
```

### 🌐 Step 2: Set Up a Virtual Environment

```bash
python -m venv venv
# Activate the environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

Your app will launch in a browser window at [http://localhost:8501](http://localhost:8501).

---

## 🎯 Features

- **Interactive Clustering:** Select the number of clusters and view customer groups in real time.
- **Preprocessing Pipeline:** Scales numerical data and encodes categorical variables.
- **Optimal Cluster Finder:** Implements both Elbow Method and Silhouette Score for evaluating clusters.
- **Visual Insights:**
  - 2D cluster scatterplots
  - Cluster centroids overlay
  - Elbow & Silhouette evaluation (optionally addable)

---

## 🗃 Data

- **Dataset:** `mall_customers.csv` includes the following fields:
  - `Customer_ID`
  - `Gender`
  - `Age`
  - `Annual_Income`
  - `Spending_Score`

---

## 📈 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn (KMeans Clustering)
- Matplotlib & Seaborn
- Streamlit

---

## 🌐 Deployment

Deploy on [Streamlit Cloud](https://streamlit.io/cloud) by connecting your GitHub repo and setting `app.py` as the entry point.

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 🙌 Contribution

Feel free to fork this project, suggest improvements, or add new features. Pull requests and contributions are always welcome!

---

## ✉️ Contact

- **Author:** [Your Name Here]
- **GitHub:** [https://github.com/just-sree](https://github.com/just-sree)
```
