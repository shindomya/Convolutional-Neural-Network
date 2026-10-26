# FeedForward Neural Network

A FeedForward neural network I built for school which uses the MNIST data set to recognize numbers drawn on a 28*28 background.

## Features & Components

* **Pre-trained Weights:** Ready to use out of the box without manual training.
* **train.py:** Trains the network from scratch and exports the optimized weights and biases.
* **test.py:** Picks a random image from the dataset and predicts the digit.
* **complete_test.py:** Runs an evaluation across all 10,000 test cases and outputs the overall accuracy.

### Run a Single Test
To pick a random image and see the network's prediction:
```bash
python test.py
```

### Evaluate Accuracy
To test the model against the entire 10,000-image evaluation set:
```bash
python complete_test.py
```

### Train from Scratch
If you want to re-train the network and overwrite the current weights:
```bash
python train.py
```
