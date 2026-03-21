import pandas as pd
df = pd.read_csv("dataset.csv")

# Load models
sbert = SentenceTransformer('all-MiniLM-L6-v2')
fasttext_model = fasttext.load_model("cc.en.300.bin")

# Train Doc2Vec
texts = df['student_answer'].tolist()
tagged = [TaggedDocument(words=t.split(), tags=[i]) for i, t in enumerate(texts)]

doc2vec_model = Doc2Vec(tagged, vector_size=100, epochs=40)
doc2vec_model.save("doc2vec.model")

# Feature extraction
features = []
for text in texts:
    sbert_vec = sbert.encode(text)
    doc_vec = doc2vec_model.infer_vector(text.split())
    fast_vec = fasttext_model.get_sentence_vector(text)

    final_vec = np.concatenate([sbert_vec, doc_vec, fast_vec])
    features.append(final_vec)

X = np.array(features)
y = df['score'].values

# PCA
pca = PCA(n_components=100)
X = pca.fit_transform(X)

# Train model
model = XGBRegressor()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

pickle.dump(pca, open("pca.pkl", "wb"))

# =========================
# Evaluation Metrics + Advanced Features
# =========================
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Predictions (Hybrid)
preds = model.predict(X)

mae = mean_absolute_error(y, preds)
r2 = r2_score(y, preds)

print("MAE:", mae)
print("R2 Score:", r2)

# Accuracy
preds_rounded = np.round(preds)
accuracy = (preds_rounded == y).mean()
print("Accuracy:", accuracy)

# =========================
# SBERT ONLY MODEL (for comparison)
# =========================
sbert_features = [sbert.encode(text) for text in texts]
sbert_features = np.array(sbert_features)

sbert_model = XGBRegressor()
sbert_model.fit(sbert_features, y)
sbert_preds = sbert_model.predict(sbert_features)

# =========================
# Comparison Graph
# =========================
plt.figure()
plt.scatter(y, preds, label="Hybrid")
plt.scatter(y, sbert_preds, label="SBERT Only")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.legend()
plt.title("Hybrid vs SBERT Performance")
plt.savefig("comparison_graph.png")
plt.show()

# =========================
# Confusion Matrix (Classification)
# =========================
y_class = y.astype(int)
preds_class = np.round(preds).astype(int)

cm = confusion_matrix(y_class, preds_class)
print("Confusion Matrix:
", cm)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# Save Results CSV
# =========================
results_df = pd.DataFrame({
    "Actual": y,
    "Predicted": preds,
    "Predicted_Rounded": preds_rounded
})

results_df.to_csv("results.csv", index=False)

print("Results saved to results.csv")

# =========================
# Graph for Paper
# =========================
plt.figure()
plt.scatter(y, preds)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores")
plt.savefig("results_graph.png")
plt.show()

print("Training complete!")
# =========================
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Predictions
preds = model.predict(X)

mae = mean_absolute_error(y, preds)
r2 = r2_score(y, preds)

print("MAE:", mae)
print("R2 Score:", r2)

# =========================
# Accuracy (Rounded Score Match)
# =========================
preds_rounded = np.round(preds)
accuracy = (preds_rounded == y).mean()
print("Accuracy:", accuracy)

# =========================
# Graph for Paper
# =========================
plt.figure()
plt.scatter(y, preds)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores")
plt.savefig("results_graph.png")
plt.show()

print("Training complete!")