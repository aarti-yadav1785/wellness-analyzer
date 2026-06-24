import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv('data/wellness_data.csv')
print(f"✅ Data loaded: {df.shape}")

features = ['sleep', 'exercise', 'screentime',
            'sunlight', 'meal_regularity',
            'enjoyable', 'loneliness',
            'understood', 'rumination',
            'phq_total', 'gad_total']

X = df[features]
y = df['risk_level']

# ==========================================
# SCALER
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 1. K-MEANS CLUSTERING
# ==========================================
print("\n--- K-Means Clustering ---")

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = df.groupby('cluster')[
    ['phq_total', 'gad_total',
     'sleep', 'loneliness']].mean().round(1)
print(cluster_summary)

# Cluster names
cluster_means = df.groupby('cluster')['phq_total'].mean()
sorted_clusters = cluster_means.sort_values()
cluster_name_map = {
    sorted_clusters.index[0]: 'Low Risk Group',
    sorted_clusters.index[1]: 'Medium Risk Group',
    sorted_clusters.index[2]: 'High Risk Group'
}
print(f"\nCluster Names: {cluster_name_map}")

# Graph — save karo
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
labels = ['Low Risk', 'Medium Risk', 'High Risk']
for i, (cluster_id, name) in enumerate(
        cluster_name_map.items()):
    mask = df['cluster'] == cluster_id
    plt.scatter(
        df[mask]['phq_total'],
        df[mask]['gad_total'],
        c=colors[i], label=name,
        alpha=0.6, s=60
    )
plt.xlabel('PHQ-9 Score (Depression)')
plt.ylabel('GAD-7 Score (Anxiety)')
plt.title('Wellness Risk Clusters — K-Means')
plt.legend()
plt.tight_layout()
os.makedirs('static', exist_ok=True)
plt.savefig('static/kmeans_wellness.png')
plt.close()
print("✅ K-Means done!")

# ==========================================
# 2. DECISION TREE
# ==========================================
print("\n--- Decision Tree ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(
    max_depth=4, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"✅ Decision Tree Accuracy: {dt_acc*100:.1f}%")

# Feature importance
importance = dt.feature_importances_
feat_imp = pd.Series(importance,
           index=features).sort_values(ascending=False)
print("\nTop 5 Important Features:")
print(feat_imp.head())

# Graph
plt.figure(figsize=(8, 5))
feat_imp.head(6).plot(kind='bar',
    color=['#0f3460', '#e94560', '#667eea',
           '#764ba2', '#f093fb', '#4facfe'])
plt.title('Which Factors Matter Most?')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/feature_importance_wellness.png')
plt.close()
print("✅ Decision Tree done!")

# ==========================================
# 3. KNN
# ==========================================
print("\n--- KNN ---")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)
knn_acc = accuracy_score(y_test,
          knn.predict(X_test))
print(f"✅ KNN Accuracy: {knn_acc*100:.1f}%")

# ==========================================
# 4. SVM
# ==========================================
print("\n--- SVM ---")

svm = SVC(kernel='rbf', C=10,
          gamma='scale', random_state=42,
          probability=True)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test,
          svm.predict(X_test))
print(f"✅ SVM Accuracy: {svm_acc*100:.1f}%")

# ==========================================
# 5. LINEAR REGRESSION
# ==========================================
print("\n--- Linear Regression ---")

# PHQ score se GAD score predict karo
X_lr = df[['phq_total', 'sleep',
           'loneliness', 'rumination']]
y_lr = df['gad_total']

lr = LinearRegression()
lr.fit(X_lr, y_lr)

# Coefficients
print("Coefficients:")
for feat, coef in zip(
        ['phq_total', 'sleep', 'loneliness',
         'rumination'], lr.coef_):
    print(f"  {feat}: {coef:.3f}")
print(f"✅ Linear Regression done!")

# ==========================================
# SAVE ALL MODELS
# ==========================================
print("\n--- Saving Models ---")

os.makedirs('models', exist_ok=True)

with open('models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('models/dt.pkl', 'wb') as f:
    pickle.dump(dt, f)

with open('models/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('models/svm.pkl', 'wb') as f:
    pickle.dump(svm, f)

with open('models/lr.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/cluster_name_map.pkl', 'wb') as f:
    pickle.dump(cluster_name_map, f)

print("✅ All models saved in models/ folder!")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*45)
print("📊 ML MODELS SUMMARY")
print("="*45)
print(f"K-Means    → 3 risk clusters ✅")
print(f"Decision Tree → {dt_acc*100:.1f}% accuracy ✅")
print(f"KNN           → {knn_acc*100:.1f}% accuracy ✅")
print(f"SVM           → {svm_acc*100:.1f}% accuracy ✅")
print(f"Linear Reg → PHQ → GAD prediction ✅")
print("="*45)