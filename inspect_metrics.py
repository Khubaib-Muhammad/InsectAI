
from utils.model_loader import load_model

models = [
    "CNN Version 1",
    "CNN Version 3"
]

for model_name in models:
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    model = load_model(model_name)

    for layer in model.layers:
        print(
            f"{layer.name:35} "
            f"{layer.__class__.__name__:25} "
            f"{layer.output.shape}"
        )
