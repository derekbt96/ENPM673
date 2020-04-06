import cv2
import numpy as np
from scipy.stats import multivariate_normal


class GMM(object):
    
    # For training
    _MAX_ITERATIONS = 600
    _EPSILON = 0.0001
    
    def __init__(self,
                 k,
                 means,
                 covariances,
                 weights,
                 median=None,
                 color=None,
                 threshold=0.7,
                 train=False,
                 train_data=None):

        self._k = k
        self._threshold = threshold
        self._color = color

        if train:
            self._train = train_data
            self._e = self._EPSILON
            self._num_samples = self._train.shape[0]

            # Thetas
            self._rs = np.zeros((self._num_samples, self._k))

            # Evenly weighted to start
            self._pis = [1./self._k] * self._k

            # Generate random starting covariance matrices
            self._sigmas = []
            for i in range(self._k):
                self._sigmas.append(gen_random_pos_def(3, 80))

            # Choose random points from the training set for means
            idxs = np.random.choice(self._num_samples, self._k, False)
            self._means = np.array(self._train[idxs, :], dtype=np.float)

            self.hooray, self.iters, self._means, self._covariances, self._weights = self._train_gmm()
            
            # Now that the data has been fit, compute the likelihoods for the training data
            sh = train_data.shape
            new_data = np.reshape(train_data, (sh[0], 1, 3))
            med = self.predict_proba(new_data)
            # self._median = np.max(np.mean(med, axis=0))
            self._median = np.max(np.mean(med, axis=1))

        else:
            self._means = means
            self._covariances = covariances
            self._weights = weights
            self._median = median

    def save_params(self, prefix):
        np.save(prefix + "median.npy", self._median)
        np.save(prefix + "mean.npy", self._means)
        np.save(prefix + "covars.npy", self._covariances)
        np.save(prefix + "weights.npy", self._weights)

    def load_params(self, prefix):
        self._median = np.load(prefix + "median.npy")
        self._means = np.load(prefix + "mean.npy")
        self._covariances = np.load(prefix + "covars.npy")
        self._weights = np.load(prefix + "weights.npy")

    def apply(self, img_s):
        if self._color:
            img_s = cv2.cvtColor(img_s, self._color)

        size = img_s.shape
        y = size[0]
        x = size[1]

        # Put all pixels in a row
        pixels = x * y
        test = np.reshape(img_s, (pixels, 3))

        likelihoods = self.predict_proba(test)

        # Reshape into image format
        loglikes = np.sum(likelihoods, axis=1)
        loglike_img = np.reshape(loglikes, (y, x))

        # Find points above threshold
        out_image = np.zeros((y, x), dtype=np.uint8)
        out_image[loglike_img > self._median*self._threshold] = 255
        return out_image

    def predict_proba(self, data):
        # Placeholders
        likelihoods = np.zeros((data.shape[0], self._k))

        # Compute likelihoods
        for i in range(self._k):
            normal = multivariate_normal.pdf(data,
                                             self._means[i],
                                             self._covariances[i],
                                             allow_singular=True)
            pd = self._weights[i] * normal
            likelihoods[:, i] = pd
        return likelihoods

    def _train_gmm(self):

        # https://cmsc426spring2019.github.io/colorseg/

        iter = 0
        last_log_like = 0.
        print "Starting gmm training..."
        while True:
            iter += 1

            # Compute pdfs for each mean and covariance
            for i in range(self._k):
                # Using scipy to compute gaussian probs for training data, for some reason need to
                # allow for singular covariances and it seems to work
                # Posterior Distribution using Bayes Rule
                normal = multivariate_normal.pdf(self._train,
                                                 self._means[i],
                                                 self._sigmas[i],
                                                 allow_singular=True)
                
                # Adjust by weight
                pd = self._pis[i] * normal
                
                # Update
                self._rs[:, i] = pd

            # Compute log likelihood
            tmp = np.sum(self._rs, axis=1)
            log_like = np.sum(np.log(tmp))

            # Have we converged?
            if iter > 1:
                print "iteration: %s     ||mu - mu_1|| = %s" % (iter, log_like - last_log_like)
                if np.abs(log_like - last_log_like) < self._e:
                    print "We did it!"
                    hooray = True
                    break

            # Have we failed?
            if iter == self._MAX_ITERATIONS:
                print "We didn't do it :("
                hooray = False
                break

            # Add the new likelihoods
            last_log_like = log_like

            # Iterate on means and covariances
            denom = np.sum(self._rs, axis=1)
            self._rs = (self._rs.T / denom).T
            nks = np.sum(self._rs, axis=0)
            for i in range(self._k):
                # Update weights
                self._pis[i] = 1. / self._num_samples * nks[i]

                # Mean
                self._means[i] = 1. / nks[i] * np.sum(self._rs[:, i] * np.transpose(self._train), axis=1)

                # Covariance
                tmp = np.matrix(self._train - self._means[i])
                self._sigmas[i] = np.array(1 / nks[i] * np.dot(np.multiply(tmp.T, self._rs[:, i]), tmp))

        # Return success, means, covars, and weights
        return hooray, iter, self._means, self._sigmas, self._pis
