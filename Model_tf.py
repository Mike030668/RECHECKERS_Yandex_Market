import tensorflow as tf

def create_classmodel( 
        num_classes,
        img_shape = (64,64,3),
        intersept_shape = (64,64,3),
        segment_shape =(256, 256,3) ,
        levels = [16, 32, 64, 128, 256, 512, 728] ,
        out_activ = 'softmax'          
):

    input_image = tf.keras.layers.Input(
        shape=img_shape,
        dtype=tf.float32,
        name='image'
    )
    input_insept = tf.keras.layers.Input(
        shape= intersept_shape,
        dtype=tf.float32,
        name='interseptions'
    )
    segment_logs = tf.keras.layers.Input(
        shape= segment_shape,
        dtype=tf.float32,
        name='segment_logits'
    )

    lay = tf.keras.layers

    img = lay.Conv2D(32, 3, padding="same")(input_image)
    img = lay.BatchNormalization()(img)
    img = lay.Activation("relu")(img)

    intsp = lay.Conv2D(32, 3,   padding="same")(input_insept)
    intsp = lay.BatchNormalization()(intsp)
    intsp = lay.Activation("relu")(intsp)

    conc = lay.concatenate([img, intsp], axis=-1)

    conc = lay.Conv2D(128, 3, padding="same")(conc)
    conc = lay.BatchNormalization()(conc)
    conc = lay.Activation("relu")(conc)

    seg = lay.Conv2D(128, 3, padding="same")(segment_logs)
    seg = lay.BatchNormalization()(seg)
    seg = lay.Activation("relu")(seg)
    seg = lay.MaxPooling2D(2, strides=2, padding="same")(seg)
    seg = lay.Conv2D(128, 1, padding="same")(seg)
    seg = lay.BatchNormalization()(seg)
    seg = lay.Activation("relu")(seg)
    seg = lay.MaxPooling2D(2, strides=2, padding="same")(seg)
    x = lay.concatenate([conc, seg], axis=-1)

    previous_block_activation = x

    for size in levels: #
        x = lay.Activation("relu")(x)
        x = lay.SeparableConv2D(size, 3, padding="same")(x)
        x = lay.BatchNormalization()(x)

        x = lay.Activation("relu")(x)
        x = lay.SeparableConv2D(size, 3, padding="same")(x)
        x = lay.BatchNormalization()(x)

        x = lay.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = lay.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = lay.add([x, residual])

        previous_block_activation = x

    x = lay.SeparableConv2D(1024, 3, padding="same")(x)
    x = lay.BatchNormalization()(x)
    x = lay.Activation("relu")(x)
    x = lay.GlobalAveragePooling2D()(x)
    x = lay.Dropout(0.5)(x)

    outputs = lay.Dense(num_classes, activation= out_activ)(x)

    return tf.keras.Model([input_image, input_insept, segment_logs], outputs)

