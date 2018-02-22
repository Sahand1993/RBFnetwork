class RBF_2D(object):
    """RBF network that takes 2-dimensional inputs and produces 2-dimensional outputs."""
    def __init__(self, nodes = 5, w1 = None, wo = None, std = 1 , eta = 0.01):
        self.nodes = nodes
        # The first layer weights (means)
        if w1:
            if not w1.shape == (self.nodes, 2):
                raise ValueError("First layer weight dimension do not match the number of nodes")
            self.w1 = w1
        else:
            self.w1 = np.random.random((self.nodes, 2))
            
        # The output weights
        if wo:
            self.wo = wo
        else:
            self.wo = np.random.normal(0,0.1,(self.nodes,2))
        # The standard deviations
        if std:
            self.stds = np.ones(self.nodes)*std
        else:
            self.stds = np.ones(self.odes)
        # The learning rate
        self.eta = eta

    def fi(self, x, means, std):
        fi = np.exp( -( (x[0]-means[0])**2 + (x[1]-means[1])**2 ) / (2*std**2) )
        return fi
    
    def transfers(self, x):
        """Calculates the vector of RBF functions for each node for an input vector X"""
        transfers = []
        for i in range(self.nodes):
            means = self.w1[i]
            std = self.stds[i]
            transfers.append(self.fi(x, means, std))
        return array(transfers)

    def transfers_batch(self, X):
        """Calculates the matrix of RBF functions for each node for an input matrix X"""
        fi_matr = []
        
        for x in X:
            fi_matr.append(self.transfers(x))

        return array(fi_matr)
        
    def output(self, x):
        """Takes one 2D input X and returns the output of the network"""
        if not x.shape == (2,):
            raise ValueError("Incorrect shape of input vector: {0}".format(X.shape))
        transfers = self.transfers(x)
        output = np.dot(transfers, self.wo)
        return output
    
    def outputs(self, X):
        if not X.shape[1] == 2:
            raise ValueError("Incorrect shape of input matrix. Must have 2 columns.")
        transfers = self.transfers_batch(X)
        outputs = np.dot(transfers, self.wo)
        return outputs
    
    def train_batch(self, X, Y):
        fi_matr = self.transfers_batch(X)
        a = np.dot(fi_matr.T, fi_matr)
        b = np.dot(fi_matr.T, Y)
        self.wo = np.linalg.lstsq(a,b,rcond=None)[0]
    
    def avg_error_norm(self, X, Y):
        outputs = self.outputs(X)
        errors = np.abs(outputs-Y)
        error_norms = np.linalg.norm(errors, axis=1)
        avg_error_norm = np.mean(error_norms)
        return avg_error_norm
        
    def init_means_comp(self, X):
        """places the means in w1 out according to the CL algorithm with leaky learning"""
        epochs = 500
        eta = 0.1
        h = self.nodes
        dists = {}
        means_dict = {}
        i=0
        for mean in self.w1:
            means_dict[i] = mean
            i+=1
        for i in range(epochs):
            if i%10 == 0 and not i == 0:
                h -= 2
                if h<=0:
                    h = 1
            for x in X:                
                for i, mean in means_dict.items():
                    dists[i] = x-mean
                sorted_dists = sorted(dists.items(), key=lambda x: abs(np.linalg.norm(x[1])))
                closest_h = sorted_dists[0:h if h<=self.nodes else self.nodes]
                i = 0
                for key, diff in closest_h:
                    delta_w = diff*eta/(1+1*i)
                    means_dict[key] += delta_w
                    i+=1
                
        self.w1 = array(list(map(lambda x: x[1], means_dict.items())))
        return means_dict
