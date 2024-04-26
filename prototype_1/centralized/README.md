# Centralized Model Training

Training a **Multilayer Perceptron (MLP)** using the centralized training set. You can find more information about the training set in the `datasets/` folder within the root directory. The final model is then evaluated on the test set from each client.

- The script `main.py` contains the code for training the model and evaluating it on the four clients. This script can also train multiple models. After training, all metrics (e.g., accuracy, precision, recall) from each client are saved in a `metrics.json` file within the `metrics/` folder.

    ```sh
    # Run from the project root dir
    python -m prototype_1.centralized.main --num-models 1
    ```
