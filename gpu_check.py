import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)

if not gpus:
    raise SystemExit("TensorFlow does not see a GPU.")

# Ask TensorFlow to fail instead of silently moving this test to the CPU.
tf.config.set_soft_device_placement(False)

with tf.device("/GPU:0"):
    a = tf.random.normal([2048, 2048])
    result = tf.matmul(a, a)

print("Test calculation ran on:", result.device)