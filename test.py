import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))

test_input = np.array([
    [1, "female", 25, 80, "C", "C", 1],
    [3, "male", 30, 8, "S", "E", 0],
    [2, "female", 40, 25, "S", "B", 2],
    [3, "male", 60, 10, "S", "T", 0]
])

print(pipe.predict(test_input))

