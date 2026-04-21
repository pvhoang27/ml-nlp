import scipy.linalg as ln
import numpy as np
from underthesea import word_tokenize

sentence = '''Khoa học dữ liệu là một lĩnh vực đòi hỏi kiến thức về toán và lập trình.
Tôi rất yêu thích Khoa học dữ liệu.'''
token = word_tokenize(sentence)
print('tokenization of sentences: ', token)