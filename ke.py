import numpy as np
import gensim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ====== DATA ======
texts = [
    "hôm nay trời đẹp",
    "tôi ăn cơm",
    "chúng ta học machine learning",
    "cơm rất ngon",
    "tôi thích ăn cơm"
]

labels = [0, 1, 2, 1, 1]  # ví dụ label

# ====== TOKENIZER ======
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, padding='post')

# one-hot label
y = to_categorical(labels)

# ====== TRAIN TEST SPLIT ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ====== TRAIN WORD2VEC ======
texts_tokenized = [sentence.split() for sentence in texts]
word_model = gensim.models.Word2Vec(
    sentences=texts_tokenized,
    vector_size=300,
    min_count=1,
    epochs=10
)

# ====== EMBEDDING MATRIX ======
embedding_dim = 300
word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

for word, i in word_index.items():
    if word in word_model.wv:
        embedding_matrix[i] = word_model.wv[word]
    else:
        embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

# ====== MODEL ======
model = Sequential()
model.add(
    Embedding(
        input_dim=len(word_index) + 1,
        output_dim=embedding_dim,
        input_length=X.shape[1],
        weights=[embedding_matrix],
        trainable=False,
    )
)

model.add(LSTM(64))
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# ====== TRAIN ======
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)