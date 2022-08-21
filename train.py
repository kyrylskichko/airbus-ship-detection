import tensorflow as tf
import keras.metrics as metrics

from utils import *
from config import batch_size, train_len
from model import model


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              metrics=['accuracy'])

epoch = 10

loss_f = losses.BinaryCrossentropy()

train_acc_metric = metrics.Accuracy()
val_acc_metric = metrics.Accuracy()

for e in range(epoch):
    step = 0
    for x, y in train_data(batch_size, train_len):
        step += 1
        with tf.GradientTape() as tape:
            logits = model(x)
            loss_value = loss_f(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y, logits)
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )

    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_val, y_val in val_data(batch_size, train_len):
        val_logits = model(x_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    model.save('models/')








