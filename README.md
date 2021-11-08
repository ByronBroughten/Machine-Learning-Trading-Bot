### Temporal Convnet Trading Bot

## Description:

Uses stock market data to train a temporal convolutional neural network to predict whether a given stock's price will increase or decrease at a defined time horizon given the stock's candlestick data (open, high, low, close) over a preceding timespan.

Divides data into three datasets: training, validation, and test. Ensures that training and validation datasets are shuffled and that the test dataset represents a later time period than the other two sets. The data is normalized and standardized based on the training dataset.

Enables multiple concurrent training sessions and implements random search to explore the efficacy of hyperparameter value combinations. The loss and accuracy of training, validating, and testing are viewable via tensorboard.

Enables stock trading through the TD Ameritrade API.

## Technology used

- PyTorch - Creates the model and datasets and implements model training.
- SciPy - Used to preprocess data.
- Zarr - Stores tensors in files to be loaded into memory in batches for training.

## Built With:

- VS Code - https://code.visualstudio.com/

## Contact:

For more information about the application, please don't hesitate to reach out.

Byron Broughten

- [GitHub](https://github.com/ByronBroughten)