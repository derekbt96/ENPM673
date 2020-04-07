import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import color_mask, color_data
from scipy.stats import multivariate_normal




def main():
    
    num_gaussians = 3
    np.random.seed(1)
    rand_data = np.random.random((2,num_gaussians))
    means = rand_data[0,:]*10.0
    covs = rand_data[1,:]*3.0
    
    # print(means)
    # print(covs)

    train = np.array([])
    for k in range(num_gaussians):
        train = np.append(train,np.random.randn(50,1)*covs[k]+means[k])

    x = np.linspace(-4,12,161)
    # x_arr = np.array([x,x,x])
    # print(x_arr.shape)

    y1 = multivariate_normal.pdf(x,means[0],covs[0])
    y2 = multivariate_normal.pdf(x,means[1],covs[1])
    y3 = multivariate_normal.pdf(x,means[2],covs[2])
    y = y1+y2+y3
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.hist(train, bins=100, range=(-4, 12), fc='b', ec='b')
    
    # plt.plot(x,y1)
    # plt.plot(x,y2)
    # plt.plot(x,y3)
    # plt.show()
    

    train = np.vstack([train.copy(),train.copy(),train.copy()])
    train = train.T
    k = num_gaussians
    e = .0001
    num_samples = train.shape[0]
    MAX_ITERATIONS = 500

    # Thetas
    rs = np.zeros((num_samples, k))

    # Evenly weighted to start
    pis = [1./k] * k

    # Generate random starting covariance matrices
    sigmas = []
    for i in range(k):
        mat = 80*np.random.rand(3, 3)
        sigmas.append(np.dot(mat, mat.transpose()))

    # Choose random points from the training set for means
    idxs = np.random.choice(num_samples, k, False)
    means = np.array(train[idxs, :], dtype=np.float)

    
    # self.hooray, self.iters, means, covariances, weights = train_gmm()

    
    iter = 0
    last_log_like = 0.
    print("Starting gmm training...")
    while True:
        iter += 1

        # Compute pdfs for each mean and covariance
        for i in range(k):
            # Using scipy to compute gaussian probs for training data, for some reason need to
            # allow for singular covariances and it seems to work
            # Posterior Distribution using Bayes Rule
            normal = multivariate_normal.pdf(train,
                                             means[i],
                                             sigmas[i],
                                             allow_singular=True)
            
            # Adjust by weight
            pd = pis[i] * normal
            
            # Update
            rs[:, i] = pd

        tmp = np.sum(rs, axis=1)
        log_like = np.sum(np.log(tmp))

        if iter > 1:
            print("iteration: %s     ||mu - mu_1|| = %s" % (iter, log_like - last_log_like))
            if np.abs(log_like - last_log_like) < e:
                break

        # Have we failed?
        if iter == MAX_ITERATIONS:
            break

        # Add the new likelihoods
        last_log_like = log_like

        # Iterate on means and covariances
        denom = np.sum(rs, axis=1)
        rs = (rs.T / denom).T
        nks = np.sum(rs, axis=0)
        for i in range(k):
            # Update weights
            pis[i] = 1. / num_samples * nks[i]

            # Mean
            means[i] = 1. / nks[i] * np.sum(rs[:, i] * np.transpose(train), axis=1)

            # Covariance
            tmp = np.matrix(train - means[i])
            sigmas[i] = np.array(1 / nks[i] * np.dot(np.multiply(tmp.T, rs[:, i]), tmp))


    



    # Now that the data has been fit, compute the likelihoods for the training data
    # sh = train_data.shape
    # new_data = np.reshape(train_data, (sh[0], 1, 3))
    # med = self.predict_proba(new_data)
    # median = np.max(np.mean(med, axis=0))
    # median = np.max(np.mean(med, axis=1))
    # print(sigmas[0].shape)
    
    # print(sigmas)
    # print(means)
    yn1 = multivariate_normal.pdf(x,means[0,0],sigmas[0][0,0])
    yn2 = multivariate_normal.pdf(x,means[1,0],sigmas[1][1,1])
    yn3 = multivariate_normal.pdf(x,means[2,0],sigmas[2][2,2])
    yn = yn1+yn2+yn3


     # fig = plt.figure()
    # plt.hist(train, bins=100, range=(-5, 15), fc='b', ec='b')
    plt.subplot(2,1,2)
    plt.plot(x,y,'r--')
    plt.plot(x,yn,'b',linewidth=2)
    plt.show()
main()
