"""
xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
"""
import tensorflow as tf

# a = tf.convert_to_tensor([[1, 2, 3],
#                           [1, 2, 3]])
# b = tf.convert_to_tensor([[2, 2, 3],
#                           [2, 2, 3]])
# c = tf.convert_to_tensor([[3, 3, 3],
#                           [3, 3, 3]])
d = tf.constant(1, shape=(2, 3, 4))
print(d)
e = tf.reshape(tensor=d, shape=(2, -1))
f = tf.reshape(d,shape=(-1,d.shape[2]*d.shape[1]))
print(e)

print(f)
