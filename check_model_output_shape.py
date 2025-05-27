import tensorflow as tf

model_path = 'lstm_trained_model.h5'
model = tf.keras.models.load_model(model_path)
print(f"Model output shape: {model.output_shape}")
