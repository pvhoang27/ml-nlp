# Danh sách stopword mẫu, bạn có thể bổ sung thêm các từ cần loại bỏ
stopword = set(['từ', 'và', 'là', 'các', 'một', 'những', 'khởi', 'nghiệp'])

def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)

# Câu ví dụ
sentence = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'
clean_sentence = remove_stopwords(sentence)
print('Sau khi loại stopword:', clean_sentence)