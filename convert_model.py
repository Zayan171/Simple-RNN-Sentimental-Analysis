import h5py
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


h5_path = "simple_rnn_imdb.h5"


# Exact original architecture
model = Sequential([
    Embedding(
        input_dim=10000,
        output_dim=128
    ),
    SimpleRNN(
        128,
        activation="relu"
    ),
    Dense(
        1,
        activation="sigmoid"
    )
])

model.build((None, 500))

print("Model created successfully.")


# Collect all datasets from H5
weights = {}

with h5py.File(h5_path, "r") as f:

    def collect_weights(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Only collect actual model weights
            if name.startswith("model_weights/"):
                weights[name] = np.array(obj)

    f.visititems(collect_weights)


print("\nWeights found:")
for name, value in weights.items():
    print(name, value.shape)


# Find weights by their ending names
embedding_weights = next(
    v for k, v in weights.items()
    if k.endswith("embedding/embedding/embeddings:0")
)

rnn_kernel = next(
    v for k, v in weights.items()
    if k.endswith("simple_rnn/simple_rnn_cell/kernel:0")
)

rnn_recurrent_kernel = next(
    v for k, v in weights.items()
    if k.endswith("simple_rnn/simple_rnn_cell/recurrent_kernel:0")
)

rnn_bias = next(
    v for k, v in weights.items()
    if k.endswith("simple_rnn/simple_rnn_cell/bias:0")
)

dense_kernel = next(
    v for k, v in weights.items()
    if k.endswith("dense/dense/kernel:0")
)

dense_bias = next(
    v for k, v in weights.items()
    if k.endswith("dense/dense/bias:0")
)

# Load original trained weights
model.get_layer("embedding").set_weights([
    embedding_weights
])

model.get_layer("simple_rnn").set_weights([
    rnn_kernel,
    rnn_recurrent_kernel,
    rnn_bias
])

model.get_layer("dense").set_weights([
    dense_kernel,
    dense_bias
])


# Save converted model
model.save("simple_rnn_imdb.keras")


print("\n===================================")
print("MODEL CONVERTED SUCCESSFULLY! ✅")
print("===================================")
print("simple_rnn_imdb.keras created.")