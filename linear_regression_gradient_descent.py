import numpy as np

# Sample data: Floor area (x) and house price (y)
x = np.array([240, 750, 1020, 1400, 1700, 2300, 2900])  # Floor area in sqft
y = np.array([7.0, 22.4, 29.7, 40.9, 52.3, 70.5, 85.1])  # Price in lac Taka

# Scale the floor area values (optional but helpful for large values)
x = x / 1000  # Scale floor area values to a smaller range

# Add a column of ones to x for the bias term (theta_0)
X = np.c_[np.ones(len(x)), x]

# Initialize theta parameters (theta_0 and theta_1)
theta = np.zeros(2)  # Initial guesses for theta_0 and theta_1
alpha = 0.1  # Further reduced learning rate
iterations = 1000  # Number of iterations

# Number of training examples
m = len(y)

# Gradient Descent
for i in range(iterations):
    # Compute the predictions
    predictions = X.dot(theta)
    
    # Compute the errors
    errors = predictions - y
    
    # Compute the gradients for theta_0 and theta_1
    gradient = (1/m) * X.T.dot(errors)
    
    # Update the parameters (theta_0 and theta_1)
    theta = theta - alpha * gradient

    # Optional: Print the cost every 100 iterations (for tracking progress)
    if i % 100 == 0:
        cost = (1/(2*m)) * np.sum(errors**2)
        print(f"Iteration {i}, Cost: {cost}, theta: {theta}")

# Final parameters (theta_0 and theta_1)
print(f"Final theta: {theta}")

# Predictions for new values (500, 1500, and 2000 sqft)
test_x = np.array([500, 1500, 2000])
test_X = np.c_[np.ones(len(test_x)), test_x / 1000]  # Don't forget to scale the test data
predictions = test_X.dot(theta)

print("Predicted prices for 500, 1500, and 2000 sqft:", predictions)
