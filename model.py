




   

from tensorflow.keras import layers, models


def create_model(input_shape=(64, 198, 1), training=False):
    inputs = layers.Input(shape=input_shape, name="data")


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

    merged = layers.Concatenate()([x_spec_feat, x_temp_feat])
    x = layers.Dense(48, activation='relu')(merged)
    x = layers.Dense(24, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name='dog_bark_prob')(x)

    return models.Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    model = create_model()
    model.summary()
