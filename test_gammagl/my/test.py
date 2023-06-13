import tensorlayerx as tlx
import os

os.environ['TL_BACKEND'] = 'mindspore'

a = tlx.convert_to_tensor([[1., 1, 1], [2, 2, 2]])
a = tlx.expand_dims(a, 0)
b = tlx.convert_to_tensor([[3., 3, 3], [4, 4, 4]])
b = tlx.expand_dims(b, 0)

c = tlx.concat([a, b], axis=0)
print(c)

e = tlx.softmax(tlx.convert_to_tensor([a, b]), 0)
print(e)
