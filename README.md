# Aircraft Predictive Maintenance

A machine learning project that predicts when equipment is likely to fail based on sensor data. The main goal is not just to predict a failure, but also to understand which sensors had the biggest influence on that prediction.

## Why I built this

A lot of machine learning projects stop once the model gives you a prediction. For something like aircraft maintenance, that is not really enough.
If a model says that a piece of equipment might fail, you would also want to know why it thinks that. This project looks at the sensor readings behind each prediction and uses feature importance to show which ones the model relied on most.

## How it works

### 1. `generate_data.py`

This script creates simulated sensor data for 1,000 machines.

The data includes air temperature, process temperature, rotational speed, torque and tool wear.

Instead of making failure happen whenever a sensor reaches a specific value, the probability of failure is calculated using the sensor readings. This adds some uncertainty to the data and makes the problem more realistic.

### 2. `main.py`

The model uses a Random Forest classifier from scikit learn.

The data is split into training and test sets, with 80 percent used for training and the remaining 20 percent used to test the model on data it has not seen before.

The model is evaluated using accuracy, precision and recall. Looking at these separately is important because failures are less common than normal operation, so accuracy on its own can be misleading.

The script also looks at feature importance and creates a chart showing which sensors had the biggest influence on the model.

## Results

The model achieved around 78 percent accuracy on the test data.

For comparison, always predicting that no failure will happen gives a baseline of around 75 percent. This means the model provides a modest improvement over t
