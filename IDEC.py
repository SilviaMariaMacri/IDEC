"""
Implementation for Improved Deep Embedded Clustering as described in paper:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.

Usage:
    Weights of Pretrained autoencoder for mnist are in './ae_weights/mnist_ae_weights.h5':
        python IDEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5
    for USPS and REUTERSIDF10K datasets
        python IDEC.py usps --update_interval 30 --ae_weights ./ae_weights/usps_ae_weights.h5
        python IDEC.py reutersidf10k --n_clusters 4 --update_interval 3 --ae_weights ./ae_weights/reutersidf10k_ae_weights.h5

Author:
    Xifeng Guo. 2017.4.30 
"""

from time import time
import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer,InputSpec,Dense,Input
from tensorflow.keras.optimizers import SGD
#from tensorflow.python.keras.utils.vis_utils import plot_model
import os
from sklearn.cluster import KMeans
from sklearn import metrics

#from DEC import cluster_acc#, ClusteringLayer, autoencoder



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
           
        self.autoencoder, self.encoder = autoencoder(self.dims, init='glorot_uniform')
        # prepare DEC model
        
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')                
        else:
            print('ae_weights must be given. E.g.')
            print('    python IDEC.py mnist --ae_weights weights.h5')
            exit()
         
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        #self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        '''
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        '''
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_dir='./results/idec'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # logging file
        import csv
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            print('ite: ',ite)
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=[p[index * self.batch_size::], x[index * self.batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1
            print('loss: ',loss)
            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5')
                #self.model.save(save_dir + '/IDEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/IDEC_model_final.h5')
        #self.model.save(save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')
        
        return y_pred



if __name__ == "__main__":
    # setting the hyper parameters 


    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist', choices=['euromds','mnist', 'usps', 'reutersidf10k'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=265, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--server', default=False)
    parser.add_argument('--ae_weights_dir')
    parser.add_argument('--ae_weights', default='ae_weights_mnist1.h5', help='This argument must be given')
    parser.add_argument('--save_dir', default='prova')
    parser.add_argument('--excl_data_dupl',default=None)
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    import json
    with open(args.save_dir+'/config.json', 'w') as file:
        json.dump(vars(args), file)

    # load dataset
    optimizer = SGD(lr=0.1, momentum=0.99)
    '''
    tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    )
    '''
    from datasets import load_mnist, load_reuters, load_usps

    if args.dataset == 'mnist':  # recommends: n_clusters=10, update_interval=140
        x, y = load_mnist()
        #optimizer = 'adam'
    elif args.dataset == 'usps':  # recommends: n_clusters=10, update_interval=30
        x, y = load_usps('data/usps')
    elif args.dataset == 'reutersidf10k':  # recommends: n_clusters=4, update_interval=3
        x, y = load_reuters('data/reuters')
    
    
    if args.dataset == 'euromds':
        import json
        x = json.load(open('euromds.json','r'))
        x = np.array(x)
        if args.excl_data_dupl == True:
            # exclude duplicate rows:
            x = np.unique(x,axis=0)
        
   
    # prepare the IDEC model
    idec = IDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=args.n_clusters, batch_size=args.batch_size)
    idec.initialize_model(ae_weights=args.ae_weights, gamma=args.gamma, optimizer=optimizer)
    #plot_model(idec.model, to_file='idec_model.png', show_shapes=True)
    idec.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred = idec.clustering(x, y=None, tol=args.tol, maxiter=args.maxiter,
                             update_interval=args.update_interval, save_dir=args.save_dir)
    print('acc:', cluster_acc(y, y_pred))
    print('clustering time: ', (time() - t0))