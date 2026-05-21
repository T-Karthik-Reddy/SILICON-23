import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(64, 198, 1), num_classes=3, training=False):
    inputs = layers.Input(shape=input_shape, name='data')
    x_env = layers.Conv2D(16, (1, 5), padding='same', activation='relu')(inputs)
    x_env = layers.MaxPooling2D(pool_size=(1, 2))(x_env)
    x_env = layers.Conv2D(24, (1, 5), padding='same', activation='relu')(x_env)
    x_env = layers.MaxPooling2D(pool_size=(1, 2))(x_env)
    x_env = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(x_env)
    x_env_feat = layers.GlobalAveragePooling2D()(x_env)
    x_spec = layers.Conv2D(16, (5, 1), padding='same', activation='relu')(inputs)
    x_spec = layers.MaxPooling2D(pool_size=(2, 1))(x_spec)
    x_spec = layers.Conv2D(24, (3, 1), padding='same', activation='relu')(x_spec)
    x_spec = layers.MaxPooling2D(pool_size=(2, 1))(x_spec)
    x_spec = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(x_spec)
    x_spec_feat = layers.GlobalAveragePooling2D()(x_spec)
    x_temp = layers.Conv2D(16, (1, 7), padding='same', activation='relu')(inputs)
    x_temp = layers.MaxPooling2D(pool_size=(1, 2))(x_temp)
    x_temp = layers.Conv2D(24, (1, 9), padding='same', activation='relu')(x_temp)
    x_temp = layers.MaxPooling2D(pool_size=(1, 2))(x_temp)
    x_temp = layers.Conv2D(32, (1, 11), padding='same', activation='relu')(x_temp)
    x_temp_feat = layers.GlobalAveragePooling2D()(x_temp)
    merged = layers.Concatenate()([x_env_feat, x_spec_feat, x_temp_feat])
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax', name='main_output')(x)
    model = models.Model(inputs=inputs, outputs=output)
    return model
if __name__ == '__main__':
    model = create_model()
    model.summary()
    print('\nModel build successful. Check summary for MVP-compatible layer shapes.')