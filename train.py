import argparse
import numpy as np
from chainer import datasets, cuda, serializers, Variable
from chainer import optimizers as O
from chainer import functions as F
from models import Generator, Discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n-z', type=int, default=62)
    parser.add_argument('--n-categorical', type=int, default=10)
    parser.add_argument('--n-continuous', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--out-generator-filename', type=str, default='./generator.model')
    return parser.parse_args()


def rnd_categorical(n, n_categorical):
    indices = np.random.randint(n_categorical, size=n)
    one_hot = np.zeros((n, n_categorical))
    one_hot[np.arange(n), indices] = 1
    return one_hot, indices


def rnd_continuous(n, n_continuous, mu=0, std=1):
    return np.random.normal(mu, std, size=(n, n_continuous))


if __name__ == '__main__':
    args = parse_args()
    gpu = args.gpu
    n_z = args.n_z
    n_categorical = args.n_categorical
    n_continuous = args.n_continuous
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    out_generator_filename = args.out_generator_filename

    # Prepare the training data
    train, _ = datasets.get_mnist(withlabel=False, ndim=2)
    train_size = train.shape[0]
    im_shape = train.shape[1:]

    # Prepare the models
    generator = Generator(n_z + n_categorical + n_continuous, im_shape)
    generator_optimizer = O.Adam(alpha=1e-3, beta1=0.5)
    generator_optimizer.setup(generator)

    discriminator = Discriminator(im_shape, n_categorical, n_continuous)
    discriminator_optimizer = O.Adam(alpha=2e-4, beta1=0.5)
    discriminator_optimizer.setup(discriminator)

    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    for epoch in range(max_epochs):
        generator_epoch_loss = np.float32(0)
        discriminator_epoch_loss = np.float32(0)

        for i in range(0, train_size, batch_size):
            # Sample noise z
            zs = xp.random.uniform(-1, 1, (batch_size, n_z)).astype(xp.float32)

            # Sample a category encoded as a one-hot vector to hopefully learn a digit
            c_categorical, categories = rnd_categorical(batch_size, n_categorical)
            c_categorical = xp.asarray(c_categorical, dtype=xp.float32)
            categories = xp.asarray(categories, dtype=xp.int32)

            # Sample continuous codes to learn rotation, thickness, etc.
            c_continuous = xp.asarray(rnd_continuous(batch_size, n_continuous), dtype=xp.float32)

            zc = xp.concatenate((zs, c_categorical, c_continuous), axis=1)

            # Forward
            x_fake = generator(zc)
            y_fake, mi = discriminator(x_fake)

            x_real = xp.zeros((batch_size, *im_shape), dtype=xp.float32)
            for xi in range(len(x_real)):
                x_real[xi] = xp.array(train[np.random.randint(train_size)])
            x_real = xp.expand_dims(x_real, 1)
            y_real, _ = discriminator(x_real)

            # Losses
            generator_loss = F.softmax_cross_entropy(y_fake, xp.ones(batch_size, dtype=xp.int32))
            discriminator_loss = F.softmax_cross_entropy(y_fake, xp.zeros(batch_size, dtype=xp.int32))
            discriminator_loss += F.softmax_cross_entropy(y_real, xp.ones(batch_size, dtype=xp.int32))

            # Mutual Information loss
            mi_categorical, mi_continuous_mean = F.split_axis(mi, [n_categorical], 1)

            # Categorical loss
            categorical_loss = F.softmax_cross_entropy(mi_categorical, categories, use_cudnn=False)

            # Continuous loss - Fix standard deviation to 1, i.e. log variance is 0
            mi_continuous_ln_var = xp.empty_like(mi_continuous_mean.data, dtype=xp.float32)
            mi_continuous_ln_var.fill(1)
            # mi_continuous_ln_var.fill(1e-6)
            continuous_loss = F.gaussian_nll(mi_continuous_mean, Variable(c_continuous), Variable(mi_continuous_ln_var))
            continuous_loss /= batch_size

            generator_loss += categorical_loss
            generator_loss += continuous_loss

            # Backprop
            generator_optimizer.zero_grads()
            generator_loss.backward()
            generator_optimizer.update()

            discriminator_optimizer.zero_grads()
            discriminator_loss.backward()
            discriminator_optimizer.update()

            generator_epoch_loss += generator_loss.data
            discriminator_epoch_loss += discriminator_loss.data

        generator_avg_loss = generator_epoch_loss / train_size
        discriminator_avg_loss = discriminator_epoch_loss / train_size

        print('Epoch {} Loss Generator: {} Loss Discriminator: {}'
              .format(epoch + 1, generator_avg_loss, discriminator_avg_loss))

    print('Saving model', out_generator_filename)
    serializers.save_hdf5(out_generator_filename, generator)

    print('Finished training')
