import gensim
from os import sep

# ====== DATA (bạn phải có dữ liệu) ======
texts = [
    "hôm nay trời đẹp",
    "tôi ăn cơm",
    "chúng ta học machine learning",
    "cơm rất ngon",
    "tôi thích ăn cơm"
]

# Tokenize đơn giản (demo)
texts_tokenized = [sentence.split() for sentence in texts]

# ====== TRAIN WORD2VEC ======
word_model = gensim.models.Word2Vec(
    sentences=texts_tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=10
)

# ====== SAVE MODEL ======
data_folder = "."
word_model.save(data_folder + sep + "word_model.model")

# ====== TEST ======
print("Vector của 'cơm':")
print(word_model.wv['cơm'])

print("\nCác từ gần nghĩa với 'cơm':")
print(word_model.wv.most_similar('cơm'))