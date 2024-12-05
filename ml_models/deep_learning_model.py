# import tensorflow as tf
# from tensorflow.keras import layers, models

def dl_model(inputs, labels):
    print("dl_model imported")
    # input = inputs
    # # Model definition
    # model = models.Sequential([
    #     layers.Input(input),  # Input layer
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(labels.shape[1] if len(labels.shape) > 1 else 1, activation='softmax' if len(labels.shape) > 1 else 'sigmoid')
    # ])

    # # Compile the model
    # model.compile(
    #     optimizer='adam',
    #     loss='categorical_crossentropy' if len(labels.shape) > 1 else 'binary_crossentropy',
    #     metrics=['accuracy']
    # )

    # # Summary
    # print(model.summary())

    # return model

# if __name__ == "main":
#     dl_model = dl_model()
#     dl_model.summary()