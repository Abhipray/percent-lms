import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot_points(x,y,save_path):
    quantiles = [.25,.5,.75]
    quants = np.zeros((5,2,3))
    mean = np.zeros((5,2))
    for i in range(5):
        z = np.nonzero(y[i,:])
        quants[i,:,:] = [np.quantile(x[i,z], quantiles), np.quantile(y[i,z], quantiles)]
        mean[i,:] = [np.average(x[i,z]), np.average(y[i,z])]
    colors = ['ko', 'ro', 'bo', 'co', 'go']
    colors_quant = [['kv', 'k>', 'k^'], ['rv', 'r>', 'r^'], ['bv', 'b>', 'b^'], ['cv', 'c>', 'c^'], ['gv', 'g>', 'g^']]
    colors_avg = ['kx', 'rx', 'bx', 'cx', 'gx']
    labels = ['lms', '%grad', '%1e-3', '%1e-4', '%1e-5']
    style = ['full', 'left', 'bottom', 'right', 'top']
    plt.figure()
    #for i in range(5):
        #plot dataset
        #plt.plot(x[i], y[i], colors[i], label=labels[i], fillstyle=style[i], mew=0)
    for i in range(5):
        #plot averages
        for j in range(3):
            plt.plot(quants[i,0,j], quants[i,1,j], colors_quant[i][j])
        plt.plot(mean[i,0], mean[i,1], colors_avg[i], label=labels[i])

    # Add labels and save to disk
    plt.legend()
    plt.xlabel('loss deviation')
    plt.ylabel('iterations')
    plt.xlim(-.0025,0.15)
    plt.savefig(save_path)


def plot(t, p, save_path, correction=1.0):
    """Plot predicted counts against true counts.

    Args:
        t: Vector of true counts (integral).
        y: Vector of predicted counts (real).
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(t, p, 'go', linewidth=2)

    # Plot y=x line (for visualizing correctness)
    l = np.arange(np.max([np.min(t), np.min(p)]), np.min([np.max(t), np.max(p)]), 0.01)
    plt.plot(l, l, c='red', linewidth=2)
    plt.xlim(np.min(t)-.1, np.max(t)+.1)
    plt.ylim(np.min(p)-.1, np.max(p)+.1)

    # Add labels and save to disk
    plt.xlabel('true counts')
    plt.ylabel('predicted counts')
    plt.savefig(save_path)
