# **Polynomial Regression from Scratch using Gradient Descent**

This project implements **Polynomial Regression** using **Gradient Descent**, built purely in Python **without any external libraries like NumPy or Scikit-learn**.

---

## **1. Understanding Polynomial Regression**

Polynomial Regression extends Linear Regression by modeling the relationship between the dependent variable **y** and independent variables **X** using a polynomial equation:

$$
y = b_0 + b_1X + b_2X^2 + b_3X^3 + \dots + b_nX^n
$$

where:

- **b₀** is the intercept (bias term).
- **b₁, b₂, ..., bₙ** are the coefficients (weights) for each feature and its polynomial terms.
- **n** is the degree of the polynomial.

The goal is to find the optimal values of these coefficients that best fit the training data.

---

## **2. Cost Function (Mean Squared Error)**

To measure how well the model fits the data, we use the **Mean Squared Error (MSE):**

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where:

- **yᵢ** is the actual value.
- **ŷᵢ** is the predicted value given by the polynomial equation.
- **n** is the total number of data points.

Our goal is to **minimize this error**.

---

## **3. Gradient Descent Optimization**

Since MSE is a function of **b₀, b₁, b₂, ..., bₙ**, we use **Gradient Descent** to iteratively update them.

### **Computing Gradients**

To minimize the error, we compute the **partial derivatives** of the MSE with respect to each coefficient:

$$
\frac{\partial MSE}{\partial b_j} = -\frac{2}{n} \sum_{i=1}^{n} X_{ij} (y_i - \hat{y}_i)
$$

for each **j = 0, 1, 2, ..., n**.

- When **j = 0**, **Xᵢⱼ = 1** (for the bias term).
- These gradients indicate how the coefficients should be updated to reduce the error.

### **Updating Coefficients**

Using the gradients, we update each coefficient **bⱼ** iteratively:

$$
b_j = b_j - \alpha \frac{\partial MSE}{\partial b_j}
$$

where **α** (learning rate) controls how large the update steps are.

This process continues **until convergence** (or for a fixed number of iterations).

---

## **4. Implementation Details**

### **Custom Polynomial Regression Class**

We build a **PolynomialRegression** class with the following methods:

- **fit(X, y)**: Trains the model using gradient descent.
- **predict(X)**: Makes predictions using the trained model.
- **Polynomial Feature Expansion** is done manually without using external libraries.

### **📌 Summary**

✅ Polynomial Regression extends Linear Regression by incorporating polynomial features.
✅ We minimize **Mean Squared Error (MSE)** to find the best coefficients.
✅ **Gradient Descent** iteratively updates the coefficients until convergence.
✅ The **learning rate (α)** determines how fast the model learns.
✅ This implementation is built **from scratch** using only core Python.

🚀 **Happy Coding!**

