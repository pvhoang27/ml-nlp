from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
import numpy as np
embedding_matrix = np.load('embedding_matrix.npy')
def cosine(x, y):
    cos_sim = np.dot(x, y)/(norm(x)*norm(y))
    return cos_sim

# Véc tơ biểu diễn từ khoa học
e0 = list(embedding_matrix[:, 0])

# Véc tơ biểu diễn từ dữ liệu
e1 = list(embedding_matrix[:, 1])

# Quan hệ tương quan ngữ nghĩa giữa từ khoa học và dữ liệu
cosine(e0, e1)

print('Cosine similarity giữa "khoa học" và "dữ liệu":', cosine(e0, e1))