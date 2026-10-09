Repositori ini berisi reproduksi kode komputasi, visualisasi grafis, pendalaman konsep teoritis, dan evaluasi hasil dari buku acuan utama:
> **John Sukup (2025)** - *scikit-learn Cookbook: Over 80 recipes for machine learning in Python with scikit-learn (Third Edition)*, Packt Publishing / O'Reilly.

---

## 📚 Tabel Navigasi Notebook (Chapter 1 – 5):

| Bab | Judul Materi | Tautan Notebook | Status |
| :---: | :--- | :---: | :---: |
| **Bab 1** | **Common Conventions and API Elements** | [Buka Notebook Bab 1](./Bab1_Common_Conventions_and_API_Elements.ipynb) | Selesai |
| **Bab 2** | **Pre-Model Workflow and Data Preprocessing** | [Buka Notebook Bab 2](./Bab2_Pre_Model_Workflow_and_Data_Prep.ipynb) | Selesai |
| **Bab 3** | **Dimensionality Reduction Techniques** | [Buka Notebook Bab 3](./Bab3_Dimensionality_Reduction_Techniques.ipynb) | Selesai |
| **Bab 4** | **Distance Metrics and Nearest Neighbors** | [Buka Notebook Bab 4](./Bab4_Distance_Metrics_and_Nearest_Neighbors.ipynb) | Selesai |
| **Bab 5** | **Linear Models and Regularization** | [Buka Notebook Bab 5](./Bab5_Linear_Models_and_Regularization.ipynb) | Selesai |

---

## 📝 Rangkuman Materi per Bab:

### Bab 1: Common Conventions and API Elements of scikit-learn
Membahas prinsip arsitektur terpadu scikit-learn (konsistensi, inspeksi, modularitas), antarmuka dasar Estimator (`fit`, `predict`, `fit_predict`), Transformer (`transform`, `fit_transform`), pembuatan komponen kustom via `BaseEstimator` dan `TransformerMixin`, otomatisasi alur kerja via `Pipeline`, inspeksi atribut berakhiran `_`, pencarian hyperparameter (`GridSearchCV`, `RandomizedSearchCV`), serta metadata estimator tags dan metadata routing.

### Bab 2: Pre-Model Workflow and Data Preprocessing
Membahas dampak kualitas data mentah, penanganan nilai hilang secara univariat (`SimpleImputer`) dan multivariat (`KNNImputer`), penskalaan fitur (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), encoding data kategorik (`OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`), pemrosesan data campuran terpadu (`ColumnTransformer`), serta rekayasa fitur polinomial dan seleksi fitur (`RFE`, `SelectFromModel`).

### Bab 3: Dimensionality Reduction Techniques
Membahas mitigasi *Curse of Dimensionality*, teknik reduksi linier tanpa pengawasan **Principal Component Analysis (PCA)** untuk retensi variansi data maksimum, teknik linier terarah **Linear Discriminant Analysis (LDA)** untuk memaksimalkan pemisahan kelas, reduksi non-linier **t-SNE** pada manifold data angka tulisan tangan, serta evaluasi komparatif akurasi model pada ruang dimensi tereduksi.

### Bab 4: Building Models with Distance Metrics and Nearest Neighbors
Membahas karakteristik model non-parametrik K-Nearest Neighbors (KNN), perbandingan geometris metrik jarak (Euclidean vs. Manhattan), penyetelan hyperparameter $k$ dan skema bobot tetangga, visualisasi kurva pembelajaran (*Learning Curves*) untuk diagnosis bias-variansi, serta evaluasi performa berbasis *Confusion Matrix Heatmap* dan *Classification Report*.

### Bab 5: Linear Models and Regularization
Membahas prinsip Ordinary Least Squares (OLS) dan kerentanannya terhadap multikolinearitas, penanganan variansi via **Ridge Regression ($L_2$)** dengan penyusutan koefisien, seleksi fitur otomatis dan ketersebaran (*sparsity*) via **Lasso Regression ($L_1$)**, penggabungan hibrida via **ElasticNet**, serta visualisasi kurva lintasan koefisien (*Coefficient Paths Plot*) terhadap kekuatan penalti $\log(\alpha)$.

---

## 🛠️ Lingkungan Komputasi & Pustaka:
* **Platform:** Google Colab / Python 3.13
* **Pustaka Utama:** `scikit-learn`, `numpy`, `pandas`, `scipy`, `matplotlib`, dan `seaborn`.
