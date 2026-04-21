import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ====== CLEAN TEXT ======
def standardize_data(row):
    row = re.sub(r"[\.,\?]+$", "", row)

    row = row.replace(",", " ").replace(".", " ") \
             .replace(";", " ").replace("“", " ") \
             .replace(":", " ").replace("”", " ") \
             .replace('"', " ").replace("'", " ") \
             .replace("!", " ").replace("?", " ") \
             .replace("-", " ")

    row = re.sub(r"\s+", " ", row)
    return row.strip().lower()


# ====== EMBEDDING TF-IDF ======
def embedding(X_train, X_test):
    global emb

    emb = TfidfVectorizer(
        min_df=1,          # FIX: dataset nhỏ → dùng 1
        max_df=1.0,        # FIX: không giới hạn %
        max_features=3000,
        sublinear_tf=True,
        ngram_range=(1,2)  # thêm bigram cho tốt hơn
    )

    emb.fit(X_train)

    X_train_tfidf = emb.transform(X_train)
    X_test_tfidf = emb.transform(X_test)

    # Save model
    joblib.dump(emb, 'tfidf.pkl')

    return X_train_tfidf, X_test_tfidf


# ====== LOAD MODEL ======
def load_embedding():
    return joblib.load('tfidf.pkl')


# ====== MAIN TEST ======
if __name__ == "__main__":
    data = [
        "Tôi thích học AI và Machine Learning",
        "AI đang thay đổi thế giới",
        "Tôi không thích học toán",
        "Machine Learning rất thú vị",
        "Toán rất khó nhưng quan trọng",
        "AI AI AI AI AI",
        "Học học học học học"
    ]

    # Clean data
    data = [standardize_data(x) for x in data]

    # Split train/test
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)

    # Embedding
    X_train_vec, X_test_vec = embedding(X_train, X_test)

    print("Train shape:", X_train_vec.shape)
    print("Test shape:", X_test_vec.shape)

    # ====== Test load model ======
    emb_loaded = load_embedding()

    new_text = ["AI rất mạnh và thú vị"]
    new_text = [standardize_data(x) for x in new_text]

    new_vec = emb_loaded.transform(new_text)

    print("New vector shape:", new_vec.shape)