def self_supervised_CIFAR10_hyperparams():
    hyperparams = dict(
        epochs=100,
        learning_rate=0.001,
        betas=(0.9, 0.999)
    )
    return hyperparams
