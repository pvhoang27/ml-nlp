from scipy.sparse import coo_matrix

# Tạo ma trận coherence dưới dạng sparse thông qua khai báo vị trí khác 0 của trục x và y
row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
col = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
data = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
X = coo_matrix((data, (row, col)), shape=(15, 15)).toarray()
print(X)