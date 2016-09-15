from chainer import Chain
from chainer import functions as F
from chainer import links as L


def lindim(shape, scale, n):
    w, h = shape
    return (w // scale) * (h // scale) * n


def convdim(shape, scale, n):
    w, h = shape
    return (n, w // scale, h // scale)


class Generator(Chain):
    def __init__(self, n_z, out_shape):
        super(Generator, self).__init__(
            fc1=L.Linear(n_z, 1024),
            fc1_bn=L.BatchNormalization(1024),

            fc2=L.Linear(1024, lindim(out_shape, 4, 128)),
            fc2_bn=L.BatchNormalization(lindim(out_shape, 4, 128)),

            dc1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc1_bn=L.BatchNormalization(64),

            dc2=L.Deconvolution2D(64, 1, 4, stride=2, pad=1)
        )
        self.out_shape = out_shape

    def __call__(self, z, test=False):
        h = F.relu(self.fc1_bn(self.fc1(z), test=test))
        h = F.relu(self.fc2_bn(self.fc2(h), test=test))
        h = F.reshape(h, (z.shape[0],) + convdim(self.out_shape, 4, 128))
        h = F.relu(self.dc1_bn(self.dc1(h), test=test))
        h = F.sigmoid(self.dc2(h))
        return h


class Discriminator(Chain):
    def __init__(self, in_shape, n_categorical, n_continuous):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(1, 64, 4, stride=2, pad=1),

            c2=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            c2_bn=L.BatchNormalization(128),

            fc1=L.Linear(lindim(in_shape, 4, 128), 1024),
            fc1_bn=L.BatchNormalization(1024),

            # Real/Fake prediction
            fc_d=L.Linear(1024, 2),

            # Mutual information reconstruction
            fc_mi1=L.Linear(1024, 128),
            fc_mi1_bn=L.BatchNormalization(128),

            fc_mi2=L.Linear(128, n_categorical + n_continuous)
        )

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c1(x), slope=0.1)
        h = F.leaky_relu(self.c2_bn(self.c2(h), test=test), slope=0.1)
        h = F.leaky_relu(self.fc1_bn(self.fc1(h), test=test), slope=0.1)

        d = self.fc_d(h)

        mi = F.leaky_relu(self.fc_mi1_bn(self.fc_mi1(h), test=test), slope=0.1)
        mi = self.fc_mi2(mi)
        return d, mi
