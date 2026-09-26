import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Dataset loaded successfully!")

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nImage normalization completed!")

print("Training pixel range:", x_train.min(), "to", x_train.max())
print("Testing pixel range:", x_test.min(), "to", x_test.max())

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nImages reshaped for CNN!")

print("Training images shape:", x_train.shape)
print("Testing images shape:", x_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

print("\nCNN model created successfully!")

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model compiled successfully!")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

print("\nCNN model training completed successfully!")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n--- Model Evaluation ---")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

predictions = model.predict(x_test, verbose=0)

predicted_labels = predictions.argmax(axis=1)

print("\nPredictions made successfully!")

print("First 10 predicted labels:", predicted_labels[:10])
print("First 10 actual labels:", y_test[:10])

plt.figure(figsize=(10, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        f"Predicted: {predicted_labels[i]}\n"
        f"Actual: {y_test[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

print("\nSample predictions saved successfully!")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig("accuracy_graph.png")
plt.show()

print("\nAccuracy graph saved successfully!")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig("loss_graph.png")
plt.show()

print("\nLoss graph saved successfully!")

model.save("handwritten_digit_cnn.keras")

print("\nTrained CNN model saved successfully!")

print("\n===================================")
print("Handwritten Character Recognition Completed!")
print("===================================")

