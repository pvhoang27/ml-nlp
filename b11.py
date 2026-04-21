import re

# ====== Hàm chuẩn hóa dữ liệu ======
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$", "", row)

    # Xóa tất cả dấu câu trong câu
    row = row.replace(",", " ").replace(".", " ") \
             .replace(";", " ").replace("“", " ") \
             .replace(":", " ").replace("”", " ") \
             .replace('"', " ").replace("'", " ") \
             .replace("!", " ").replace("?", " ") \
             .replace("-", " ")

    # Xóa nhiều khoảng trắng liên tiếp
    row = re.sub(r"\s+", " ", row)

    return row.strip()


# ====== Tokenizer đơn giản ======
def tokenize(text):
    return text.split()


# ====== Test thử ======
if __name__ == "__main__":
    text = 'Xin chào!!! Tôi đang học NLP, bạn có khỏe không??? Đây là test-case.'
    
    print("Original:", text)
    
    cleaned = standardize_data(text)
    print("Cleaned:", cleaned)
    
    tokens = tokenize(cleaned)
    print("Tokens:", tokens)