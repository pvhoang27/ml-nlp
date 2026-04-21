import re
from os import listdir, sep


def preProcess(sentences):
    # Bỏ ký tự không phải chữ/số/khoảng trắng (giữ Unicode tiếng Việt)
    text = [
        re.sub(r"([^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF\s\w]|_)+", "", sentence)
        for sentence in sentences
        if sentence and sentence.strip()
    ]

    # Chuẩn hóa các ký tự phân tách thành khoảng trắng
    text = [
        re.sub(r"([\W_]+)", " ", sentence, flags=re.UNICODE)
        for sentence in text
        if sentence and sentence.strip()
    ]

    # lowercase + trim
    text = [sentence.lower().strip() for sentence in text if sentence.strip()]
    return text


def loadData(data_folder):
    texts = []
    labels = []

    for folder in listdir(data_folder):
        if folder == ".DS_Store":
            continue

        print("Load cat:", folder)
        folder_path = data_folder + sep + folder

        for file in listdir(folder_path):
            if file == ".DS_Store":
                continue

            print("Load file:", file)
            file_path = folder_path + sep + file

            with open(file_path, "r", encoding="utf-8") as f:
                all_of_it = f.read()
                sentences = all_of_it.split(".")
                sentences = preProcess(sentences)

                texts += sentences
                labels += [folder for _ in sentences]

    return texts, labels


if __name__ == "__main__":
    data_folder = "duong_dan_toi_thu_muc_du_lieu"
    texts, labels = loadData(data_folder)

    print("Số lượng câu:", len(texts))
    print("Số lượng nhãn:", len(labels))
    print("Một số câu mẫu:", texts[:5])
    print("Nhãn tương ứng:", labels[:5])