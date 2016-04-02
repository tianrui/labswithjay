import numpy as np
def txtfile2id(filename):
"""Load raw DNA text file, converts strings to integer id 
   numpy array using the following mapping:
         A:0
         C:1
         G:2
         T:3 
  Args:
    filename: string path to the data text file.
  Returns:
    numpy array of size (num_data, sequence_length)
  """
    letter_to_id = {'A':0,'C':1,'G':2,'T':3}
    fp = open(filename)
    raw_list = fp.readlines()
    data_list = []
    for item in raw_list:
        id_list = []
        for c in item.strip():
            id_list.append(letter_to_id[c])
        data_list.append(np.array(id_list))
    return np.array(data_list)

