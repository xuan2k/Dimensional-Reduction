import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_distances(X):
    """
    Calculate pairwise Euclidean distances between data points.
    Args:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
    Returns:
        numpy.ndarray: Pairwise distance matrix of shape (n_samples, n_samples).
    """

    n_samples = X.shape[0]
    pairwise_distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distance = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            pairwise_distances[i, j] = distance
            pairwise_distances[j, i] = distance
    return pairwise_distances

def calculate_similarity_matrix(pairwise_distances, perplexity, epsilon=1e-8):

    """
    Calculate the similarity matrix using Gaussian kernel based on pairwise distances.
    Args:
        pairwise_distances (numpy.ndarray): Pairwise distance matrix of shape (n_samples, n_samples).
        perplexity (float): Perplexity value.
        epsilon (float): Small value to prevent division by zero.
    Returns:
        numpy.ndarray: Similarity matrix of shape (n_samples, n_samples).
    """

    n_samples = pairwise_distances.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            squared_diff = (pairwise_distances[i, j] ** 2)
            similarity = np.exp(-squared_diff / (2 * perplexity))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    # Normalize the similarity matrix
    sum_similarity = np.maximum(np.sum(similarity_matrix, axis=1), epsilon)
    similarity_matrix /= sum_similarity[:, np.newaxis]
    return similarity_matrix

def calculate_gradient(Y, P, Q):
    """
    Calculate the gradient of the cost function with respect to the lower-dimensional points.
    Args:
        Y (numpy.ndarray): Current embedded points of shape (n_samples, n_components).
        P (numpy.ndarray): Similarity matrix in the high-dimensional space of shape (n_samples, n_samples).
        Q (numpy.ndarray): Similarity matrix in the lower-dimensional space of shape (n_samples, n_samples).
    Returns:
        numpy.ndarray: Gradient of the cost function with respect to Y of shape (n_samples, n_components).
    """
    n_samples, _ = Y.shape
    grad = np.zeros_like(Y)
    for i in range(n_samples):
        diff = Y[i, :] - Y
        diff_squared = np.sum(diff ** 2, axis=1)
        factor = (P[i, :] - Q[i, :]) * (1 + diff_squared) ** -1
        grad[i, :] = 4 * np.dot(factor, diff)
    return grad

def sgd_update(Y, learning_rate, grad):
    """
    Update the embedded points using Stochastic Gradient Descent.
    Args:
        Y (numpy.ndarray): Current embedded points of shape (n_samples, n_components).
        learning_rate (float): Learning rate for gradient updates.
        grad (numpy.ndarray): Gradient of the cost function with respect to Y of shape (n_samples, n_components).
    Returns:
        numpy.ndarray: Updated embedded points of shape (n_samples, n_components).
    """
    return Y - learning_rate * grad

def optimize_sne(X, n_components, perplexity, n_iterations, learning_rate):
    """
    Optimize Stochastic Neighbor Embedding (SNE) to reduce the dimensionality of the data.
    Args:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        n_components (int): Number of components in the lower-dimensional space.
        perplexity (float): Perplexity value.
        n_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for gradient updates.
    Returns:
        numpy.ndarray: Embedded points in the lower-dimensional space of shape (n_samples, n_components).
    """
    n_samples, _ = X.shape

    # Step 1: Compute pairwise distances
    pairwise_distances_X = calculate_pairwise_distances(X)
    pairwise_distances_Y = calculate_pairwise_distances(Y)

    # Step 2: Initialize embedded points randomly
    Y = np.random.randn(n_samples, n_components)

    # Step 3: Optimization loop
    for iteration in range(n_iterations):
        # Step 4: Calculate similarity matrix in the high-dimensional space
        P = calculate_similarity_matrix(pairwise_distances_X, perplexity)

        # Step 5: Calculate similarity matrix in the lower-dimensional space
        Q = calculate_similarity_matrix(pairwise_distances_Y, perplexity)

        # Step 6: Calculate gradient of the cost function
        grad = calculate_gradient(Y, P, Q)

        # Step 7: Update embedded points using SGD
        Y = sgd_update(Y, learning_rate, grad)

    return Y

def t_sne(X, n_components=1, perplexity=30.0, n_iterations=1000, learning_rate=200.0, random_state=None):
    """
    t-SNE algorithm for dimensionality reduction.
    Args:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        n_components (int): Number of components in the lower-dimensional space.
        perplexity (float): Perplexity value.
        n_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for gradient updates.
        random_state (int): Seed for random number generator.
    Returns:
        numpy.ndarray: Embedded points in the lower-dimensional space of shape (n_samples, n_components).
    """
    
    if random_state is not None:
        np.random.seed(random_state)

    # Step 1: Optimize SNE
    Y = optimize_sne(X, n_components, perplexity, n_iterations, learning_rate)

    # Step 2: Apply Student's t-distribution to the embedded points
    pairwise_distances = calculate_pairwise_distances(Y)
    Q = calculate_similarity_matrix(pairwise_distances, perplexity)
    np.fill_diagonal(Q, 0) # qii = 0 as in report 
    Q /= np.sum(Q)
    Q = np.maximum(Q, 1e-12)  # Avoid division by zero
    Y = np.dot(np.sqrt(Q), Y)

    return Y