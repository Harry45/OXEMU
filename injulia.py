"""
Author: Dr. Arrykrishna Mootoovaloo
Date: 30th June 2022
Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
Description: Example of code to use in Julia.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.interpolate as itp


def interpolate(inputs: list) -> np.ndarray:
    """Function to interpolate the power spectrum along the redshift axis

    Args:
        inputs (list): x values, y values and new values of x

    Returns:
        np.ndarray: an array of the interpolated power spectra
    """

    xval, yval, xnew = np.log(inputs[0]), np.log(inputs[1]), np.log(inputs[2])

    spline = itp.splrep(xval, yval)

    ynew = itp.splev(xnew, spline)

    return np.exp(ynew)


def compute(arr1: np.ndarray, arr2: np.ndarray, hyper: np.ndarray) -> np.ndarray:
    """Compute the kernel matrix between two sets of points.

    Args:
        x1 (np.ndarray): [N x d] tensor of points.
        x2 (np.ndarray): [M x d] tensor of points.
        hyper (np.ndarray): [d+1] tensor of hyperparameters.

    Returns:
        np.ndarray: a tensor of size [N x M] containing the kernel matrix.
    """

    ndim = arr1.shape[1]

    # reshape all tensors in the right dimensions
    arr2 = arr2.reshape(-1, ndim)

    # for the hyperparameters, we have an amplitude and ndim lengthscales
    hyper = hyper.reshape(1, ndim + 1)

    # the inputs are scaled by the characteristic lengthscale
    arr1 = arr1 / np.exp(hyper[:, 1:])
    arr2 = arr2 / np.exp(hyper[:, 1:])

    # compute the pairwise distance
    term1 = np.sum(np.power(arr1, 2), 1, keepdims=True)
    term2 = 2 * np.dot(arr1, arr2.T)
    term3 = np.sum(np.power(arr2, 2), 1, keepdims=True).T
    dist = term1 - term2 + term3

    # compute the kernel
    kernel = np.exp(hyper[:, 0]) * np.exp(-0.5 * dist)

    return kernel


class TransForm(object):
    """Implements the transformation of the inputs and outputs.

    Args:
        xinputs (np.ndarray): the inputs to be transformed.
        yinputs (np.ndarray): the outputs to be transformed.
    """

    def __init__(self, xinputs: np.ndarray, yinputs: np.ndarray):

        # dimensionality of the problem
        self.ndim = xinputs.shape[1]

        # compute the covariance of the inputs (ndim x ndim)
        self.cov_train = np.cov(xinputs.T)

        # compute the Cholesky decomposition of the matrix
        self.chol_train = np.linalg.cholesky(self.cov_train)

        # compute the mean of the sample
        self.mean_train = np.mean(xinputs, axis=0).reshape(1, self.ndim)

        # transformation for the outputs (power spectrum)
        ylog = np.log(yinputs)
        self.ymean = np.mean(ylog, axis=0)
        self.ystd = np.std(ylog, axis=0)

    def x_transformation(self, point: np.ndarray) -> np.ndarray:
        """Pre-whiten the input parameters.

        Args:
            point (torch.tensor): the input parameters.

        Returns:
            torch.tensor: the pre-whitened parameters.
        """

        # ensure the point has the right dimensions
        point = point.reshape(-1, self.ndim)

        # calculate the transformed training points
        transformed = np.linalg.inv(self.chol_train) @ (point - self.mean_train).T

        return transformed.T

    def y_transformation(self, yvalues: np.ndarray) -> np.ndarray:
        """Transform the outputs.

        Args:
            yvalues (np.ndarray): the values to be transformed

        Returns:
            np.ndarray: the transformed outputs
        """
        return (yvalues - self.ymean) / self.ystd

    def inv_y_transformation(self, yvalues: np.ndarray) -> np.ndarray:
        """Transform the outputs.

        Args:
            yvalues (np.ndarray): the values to be transformed

        Returns:
            np.ndarray: the transformed outputs
        """
        return np.exp(yvalues * self.ystd + self.ymean)


@dataclass
class JuliaPredictions(TransForm):
    """Make predictions using the GPs. To replicate in Julia.

    Args:
        xinputs (np.ndarray): the inputs to the GP.
        yinputs (np.ndarray): the outputs of the GP.
        hyper (np.ndarray): the hyperparameters of the GP.
        alphas (np.ndarray): the weights of the GP.
        xtrans (bool): whether to pre-whiten the inputs.
        ytrans (bool): whether to transform the outputs.
    """

    xinputs: np.ndarray
    yinputs: np.ndarray
    hyper: np.ndarray
    alphas: np.ndarray
    xtrans: bool = field(default=True)
    ytrans: bool = field(default=True)

    def __post_init__(self):

        # number of GPs
        self.ngps = self.alphas.shape[0]

        # transformations
        TransForm.__init__(self, self.xinputs, self.yinputs)

        if self.xtrans:
            self.xtrain = TransForm.x_transformation(self, self.xinputs)

        if self.ytrans:
            self.ytrain = TransForm.y_transformation(self, self.yinputs)

    def mean_prediction(self, testpoint: np.ndarray) -> np.ndarray:
        """Compute the mean prediction of the GP.

        Args:
            testpoint (np.ndarray): the test points.

        Returns:
            np.ndarray: the mean prediction.
        """

        if self.xtrans:
            testpoint = TransForm.x_transformation(self, testpoint)

        preds = list()

        for i in range(self.ngps):
            kernel = compute(self.xtrain, testpoint, self.hyper[i]).reshape(-1)
            mean = np.dot(kernel, self.alphas[i])
            preds.append(mean)

        preds = np.array(preds)

        if self.ytrans:
            preds = TransForm.inv_y_transformation(self, preds)

        return preds
