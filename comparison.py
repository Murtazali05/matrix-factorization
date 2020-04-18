import pandas as pd

import dataset as dp
from svd import SVD
from svdpp import SVDpp

print('--------------------Matrix factorization--------------------')
train_data, test_data, data = dp.load_dataset()

table = {}

f = 10
model = SVD(train_data, f)
model.train()
table[f] = [model.test(test_data)]

model = SVDpp(train_data, f)
model.train()
table[f].append(model.test(test_data))


f = 20
model = SVD(train_data, f)
model.train()
table[f] = [model.test(test_data)]

model = SVDpp(train_data, f)
model.train()
table[f].append(model.test(test_data))


f = 50
model = SVD(train_data, f)
model.train()
table[f] = [model.test(test_data)]

model = SVDpp(train_data, f)
model.train()
table[f].append(model.test(test_data))



f = 100
model = SVD(train_data, f)
model.train()
table[f] = [model.test(test_data)]

model = SVDpp(train_data, f)
model.train()
table[f].append(model.test(test_data))



f = 200
model = SVD(train_data, f)
model.train()
table[f] = model.test(test_data)

model = SVDpp(train_data, f)
model.train()
table[f].append(model.test(test_data))





# # intialise data of lists.
# data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
#         'Age': [20, 21, 19, 18]}

# Create DataFrame
df = pd.DataFrame(table)

# Print the output.
print(df)

print('----------------------------End-----------------------------')
