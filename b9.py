
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

def txtTokenizer(texts, num_words=500):
    # Khởi tạo tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    # Huấn luyện tokenizer trên dữ liệu
    tokenizer.fit_on_texts(texts)
    # Lấy từ điển từ -> số
    word_index = tokenizer.word_index
    # Chuyển văn bản thành chuỗi số
    X = tokenizer.texts_to_sequences(texts)
    # Đệm các chuỗi về cùng độ dài
    X = pad_sequences(X)
    return tokenizer, word_index, X

# Ví dụ sử dụng:
texts = ['hôm nay chúng ta học xử lý ngôn ngữ tự nhiên']
tokenizer, word_index, X = txtTokenizer(texts)
print(X)