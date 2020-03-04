def print_keras_model_layers(model):
    for layer in model.layers:
        print(layer.name)
