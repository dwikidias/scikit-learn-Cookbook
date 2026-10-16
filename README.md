# scikit-learn Cookbook (Third Edition) - Code Reproduction & Theory Deep-Dive

* **Nama:** [Nama Lengkap Anda]
* **NIM:** [NIM Anda]
* **Kelas:** BS1TK-47-REG-G05
* **Mata Kuliah:** Pembelajaran Mesin / Machine Learning
* **Tugas:** Assignment 2 (Milestone 1 & Milestone 2)
* **Buku Referensi:** *scikit-learn Cookbook: Over 80 recipes for machine learning in Python with scikit-learn (Third Edition)* - John Sukup (Packt / O'Reilly, 2025)

---

## 📚 Tabel Navigasi Notebook (Chapter 1 – 8)

| Bab | Judul Materi | Tautan Notebook | Status Pengerjaan |
| :---: | :--- | :---: | :---: |
| **Bab 1** | **Common Conventions and API Elements** | [Buka Notebook Bab 1](./Bab1_Common_Conventions_and_API_Elements.ipynb) | Selesai (Milestone 1) |
| **Bab 2** | **Pre-Model Workflow and Data Preprocessing** | [Buka Notebook Bab 2](./Bab2_Pre_Model_Workflow_and_Data_Prep.ipynb) | Selesai (Milestone 1) |
| **Bab 3** | **Dimensionality Reduction Techniques** | [Buka Notebook Bab 3](./Bab3_Dimensionality_Reduction_Techniques.ipynb) | Selesai (Milestone 1) |
| **Bab 4** | **Distance Metrics and Nearest Neighbors** | [Buka Notebook Bab 4](./Bab4_Distance_Metrics_and_Nearest_Neighbors.ipynb) | Selesai (Milestone 1) |
| **Bab 5** | **Linear Models and Regularization** | [Buka Notebook Bab 5](./Bab5_Linear_Models_and_Regularization.ipynb) | Selesai (Milestone 1) |
| **Bab 6** | **Advanced Logistic Regression and Extensions** | [Buka Notebook Bab 6](./Bab6_Advanced_Logistic_Regression.ipynb) | Selesai (Milestone 2) |
| **Bab 7** | **Support Vector Machines and Kernel Methods** | [Buka Notebook Bab 7](./Bab7_Support_Vector_Machines.ipynb) | Selesai (Milestone 2) |
| **Bab 8** | **Tree-Based Algorithms and Ensemble Methods** | [Buka Notebook Bab 8](./Bab8_Tree_Based_Ensembles.ipynb) | Selesai (Milestone 2) |

---

## 📝 Dokumentasi Rinci Pembahasan per Bab

---

### BAB 1: COMMON CONVENTIONS AND API ELEMENTS OF SCIKIT-LEARN
Membahas arsitektur terstandarisasi yang mendasari konsistensi pustaka `scikit-learn`:
1. **Filosofi Desain:** Konsistensi antarmuka (*consistency*), kemudahan inspeksi atribut, modularitas, dan non-proliferasi objek menggunakan NumPy/Pandas.
2. **Estimator:** Pola pemanggilan method seragam `fit()` untuk pembelajaran terarah/tak-terarah, `predict()` untuk inferensi, dan `fit_predict()` untuk klastering efisien.
3. **Transformer:** Menghitung parameter via `fit()` dan menerapkan perubahan via `transform()`. Menegakkan aturan pemisahan ketat pada data uji untuk mencegah kebocoran data (*data leakage*).
4. **Custom Components:** Memperluas pustaka menggunakan `BaseEstimator` (dukungan otomatis `get_params` dan `set_params`) serta `TransformerMixin` (penyediaan otomatis `fit_transform`).
5. **Pipeline:** Merangkai pembersihan, penskalaan, dan model ke dalam satu alur kerja terpadu yang aman untuk deployment produksi.
6. **Atribut dan Konvensi:** Penggunaan *trailing underscore* (`_`) untuk menandai bobot hasil pembelajaran model (`coef_`, `intercept_`, `classes_`), serta method evaluasi baku `score()`.
7. **Hyperparameter Tuning:** Eksplorasi ruang parameter terstruktur menggunakan `GridSearchCV` dan sampling efisien menggunakan `RandomizedSearchCV`.
8. **Metadata & Tags:** Pemeriksaan kapabilitas estimator secara terprogram melalui `__sklearn_tags__()` dan perutean bobot data via *Metadata Routing*.

---

### BAB 2: PRE-MODEL WORKFLOW AND DATA PREPROCESSING
Membahas teknik penyiapan dan rekayasa data mentah sebelum pemodelan:
1. **Penanganan Missing Data:** Membedakan mekanisme data hilang (MCAR, MAR, MNAR). Membandingkan imputasi univariat median via `SimpleImputer` terhadap imputasi multivariat kontekstual berbasis tetangga terdekat via `KNNImputer`.
2. **Penskalaan Fitur (*Feature Scaling*):** Menguji karakteristik `StandardScaler` (z-score), `MinMaxScaler` (rentang $[0, 1]$), dan `RobustScaler` (berbasis Median dan IQR yang terbukti tahan terhadap pencilan ekstrem).
3. **Encoding Data Kategorik:** Mengubah fitur nominal via `OneHotEncoder(drop='first')` untuk mencegah *dummy variable trap*, memetakan fitur bertingkat via `OrdinalEncoder` dengan hierarki eksplisit, serta mengodekan target biner via `LabelEncoder`.
4. **Alur Preprocessing Campuran (`ColumnTransformer`):** Menggabungkan alur numerik dan kategorik secara paralel dalam satu DataFrame terpadu.
5. **Rekayasa dan Seleksi Fitur:** Membentuk suku non-linier dan interaksi variabel via `PolynomialFeatures`, serta memangkas dimensi fitur non-informatif via `RFE` (Recursive Feature Elimination) dan `SelectFromModel`.

---

### BAB 3: DIMENSIONALITY REDUCTION TECHNIQUES
Membahas mitigasi fenomena *Curse of Dimensionality* melalui kompresi fitur:
1. **Principal Component Analysis (PCA):** Teknik linier *unsupervised* yang memproyeksikan data ke arah variansi maksimum. Dilengkapi visualisasi **Scatter Plot 2D (PC1 vs PC2)** dan **Bar Chart Explained Variance Ratio** pada dataset Wine (mereduksi 13 fitur menjadi 2 komponen dengan retensi variansi $\approx 55.41\%$).
2. **Linear Discriminant Analysis (LDA):** Teknik linier *supervised* berbasis kriteria Fisher yang memaksimalkan pemisahan antar kelas. Dilengkapi visualisasi komparasi berdampingan **PCA vs. LDA**, membuktikan pemisahan klaster kelas yang jauh lebih tegas dan bersih pada LDA.
3. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Teknik non-linier *manifold learning* yang memetakan data berdimensi tinggi (64 piksel pada dataset Digits) menjadi pulau-pulau klaster angka 2D yang bermakna.
4. **Evaluasi Performa Model:** Validasi silang membuktikan bahwa mereduksi data menjadi 2 komponen LDA menghasilkan akurasi klasifikasi tertinggi ($\approx 98.9\%$).

---

### BAB 4: BUILDING MODELS WITH DISTANCE METRICS AND NEAREST NEIGHBORS
Membahas pemodelan non-parametrik berbasis kedekatan spasial K-Nearest Neighbors (KNN):
1. **Tinjauan Metrik Jarak:** Menghitung dan memvisualisasikan perbandingan geometris antara **Jarak Euclidean ($L_2$)** (garis lurus diagonal) dan **Jarak Manhattan ($L_1$)** (jalur kisi tegak lurus yang lebih tahan outlier).
2. **Hyperparameter Tuning:** Mengoptimalkan kombinasi nilai $k$ ganjil dan skema bobot (`uniform` vs `distance`) via `GridSearchCV`.
3. **Kurva Pembelajaran (*Learning Curves*):** Memvisualisasikan konvergensi skor data latih dan validasi silang seiring penambahan jumlah sampel untuk mendiagnosis kestabilan model.
4. **Evaluasi Komprehensif:** Menyajikan visualisasi **Confusion Matrix Heatmap** dan laporan klasifikasi (Presisi, Recall, F1-Score), mendokumentasikan akurasi pengujian $\approx 97.78\%$ pada dataset Iris.

---

### BAB 5: LINEAR MODELS AND REGULARIZATION
Membahas regresi linier dan teknik pembatasan bobot koefisien:
1. **Ordinary Least Squares (OLS) & Multikolinearitas:** Menunjukkan kerentanan OLS terhadap data dengan korelasi tinggi, dilengkapi visualisasi garis kecocokan regresi linier.
2. **Ridge Regression ($L_2$):** Menambahkan penalti kuadrat koefisien ($\alpha \sum \beta_j^2$) untuk menyusutkan koefisien mendekati nol, menstabilkan estimasi pada data multikolinearitas.
3. **Lasso Regression ($L_1$):** Menambahkan penalti nilai mutlak ($\alpha \sum \vert{}\beta_j\vert{}$) yang menghasilkan ketersebaran (*sparsity*) dengan memaksa koefisien fitur non-informatif menjadi tepat nol.
4. **Visualisasi Garis Prediksi Komparatif:** Menyajikan perbandingan visual garis OLS vs. Ridge vs. Lasso pada data uji.
5. **ElasticNet & Coefficient Paths Plot:** Menggabungkan penalti $L_1$ dan $L_2$, dilengkapi visualisasi kurva lintasan koefisien (*Coefficient Paths*) terhadap kekuatan penalti $\log(\alpha)$ untuk memperlihatkan dinamika penyusutan bobot secara dinamis.

---

### BAB 6: ADVANCED LOGISTIC REGRESSION AND EXTENSIONS
Membahas regresi logistik tingkat lanjut dan perluasan klasifikasi:
1. **Klasifikasi Biner & Confusion Matrix:** Memetakan probabilitas via fungsi sigmoid, dilengkapi visualisasi **Confusion Matrix Heatmap** pada diagnosis kanker payudara ($\text{Akurasi} \approx 97.9\%$).
2. **Strategi Multi-kelas (OvR vs. Multinomial Softmax):** Membandingkan arsitektur One-vs-Rest terhadap Multinomial, dilengkapi visualisasi **Pairplot Multivariat** sebaran fitur bunga Iris.
3. **Regularisasi & Batas Keputusan (*Decision Boundaries*):** Memvisualisasikan perbandingan bidang keputusan kontur 2D (*meshgrid*) antara penalti ketat ($C=0.05$, batas halus stabil) vs penalti longgar ($C=50.0$, batas berliku sensitif).
4. **Klasifikasi Multi-label (*Multilabel Classification*):** Menerapkan `MultiOutputClassifier` untuk memprediksi beberapa label biner sekaligus pada satu sampel data, dievaluasi melalui *Hamming Loss* dan visualisasi matriks komparasi label aktual vs prediksi.

---

### BAB 7: SUPPORT VECTOR MACHINES AND KERNEL METHODS
Membahas pengklasifikasi margin maksimal dan metode kernel:
1. **Linear SVM, Margin, & Support Vectors:** Menemukan hyperplane pemisah optimal dengan margin terluas. Dilengkapi visualisasi bidang keputusan 2D yang menyoroti garis batas margin ($w^T x + b = \pm 1$) dan titik-titik penentu *Support Vectors*.
2. **The Kernel Trick (RBF vs. Polynomial):** Memetakan data non-linier ke ruang dimensi tinggi secara implisit. Dilengkapi visualisasi perbandingan bidang batas keputusan pada dataset bulan sabit (*make_moons*), membuktikan keunggulan adaptasi kernel RBF dibanding Polinomial.
3. **Tuning Hyperparameter $C$ dan Gamma ($\gamma$):** Memvisualisasikan dampak variasi parameter $\gamma$ dalam 3 panel berdampingan ($\gamma = 0.1$ datar kaku, $\gamma = 1.0$ optimal seimbang, $\gamma = 10.0$ berliku *overfitting*), serta optimasi simultan via `GridSearchCV`.

---

### BAB 8: TREE-BASED ALGORITHMS AND ENSEMBLE METHODS
Membahas algoritma berbasis pohon dan metode ansambel:
1. **Decision Trees (Pohon Keputusan):** Membagi ruang fitur secara hierarkis berbasis kriteria ketidakmurnian Gini. Dilengkapi visualisasi diagram hierarki **Decision Tree Structure (`plot_tree`)** terpotong (`max_depth=3`) yang menyajikan aturan logika diagnosis IF-THEN secara transparan.
2. **Random Forest (Bagging):** Menggabungkan 100 pohon keputusan bootstrap independen dengan pemilihan subset fitur acak di setiap simpul. Dilengkapi visualisasi diagram batang horizontal **Top 10 Feature Importance (Mean Decrease Gini)**.
3. **Gradient Boosting (Boosting):** Pembelajaran bertahap yang mengoreksi sisaan eror secara sekuensial.
4. **Evaluasi Komparatif Ansambel:** Menyajikan perbandingan grafis akurasi 5-Fold Cross-Validation, membuktikan keunggulan signifikan Random Forest ($\approx 96.1\%$) dan Gradient Boosting ($\approx 96.3\%$) dibanding pohon tunggal ($\approx 92.4\%$).

---

## 🛠️ Lingkungan Komputasi & Pustaka
* **Platform:** Google Colab / Python 3.12+
* **Pustaka Utama:** `scikit-learn`, `numpy`, `pandas`, `scipy`, `matplotlib`, dan `seaborn`.
