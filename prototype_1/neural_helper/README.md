# MLP

Here contains the code for training and evaluate a binary classifier using a **Multilayer Perceptron (MLP)** architecture.

For now there are two MLP architectures available.

**Proximal Term Support:** The `train` function can now calculate a proximal term if a `proximal_mu` value is provided. This functionality is used within the context of the `FedProx` algorithm.

```py
...
proximal_term = calculate_proximal_term(net, global_params)
loss = (
    criterion(outputs, labels)
    + (train_config["proximal_mu"] / 2) * proximal_term
)
...
```
