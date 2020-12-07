import sys
import re
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Encoding DNA data
#   - Ordinal
#   - One-hot
#   - K-mer

def SequenceToArray(seq_str, unknown='x'):
    seq_str = re.sub('[^acgt]', unknown, seq_str.lower())
    return numpy.array(list(seq_str))

def Encode_Ordinal(seq_array, ordinal_type='float', unknown='x'):
    
    label_encoder = LabelEncoder()
    encoded_seq_array = label_encoder.fit_transform(seq_array)

    if ordinal_type == 'float':
        encoder_map = {'a': 0.25, 'c': 0.50, 'g': 0.75, 't': 1.00, unknown: 0.00}
        encoded_seq_array = []
        for item in seq_array:
            encoded_seq_array.append(encoder_map[item])

    return encoded_seq_array

def Encode_OneHot(seq_array):
    label_encoder = LabelEncoder()
    integer_encoded_seq_array = label_encoder.fit_transform(seq_array)

    encoded_seq_array = OneHotEncoder(sparse=False)
    integer_encoded_seq_array = integer_encoded_seq_array.reshape(len(integer_encoded_seq_array), 1)
    encoded_seq_array = encoded_seq_array.fit_transform(integer_encoded_seq_array)
    # encoded_seq_array = numpy.delete(encoded_seq_array, -1, 1)

    return encoded_seq_array

def Encode_KMer(seq_str, k):
    KMers = []
    seq_str = seq_str.lower()
    for i in range(len(seq_str) - k + 1):
        KMers.append(seq_str[i:i+k])

    return KMers

test_seq = 'ACGTN'
print(Encode_Ordinal(SequenceToArray(test_seq)))
print(Encode_OneHot(SequenceToArray(test_seq)))
print(Encode_KMer(test_seq, 2))