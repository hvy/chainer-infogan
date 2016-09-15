# InfoGAN

Chainer implementations of *InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets*  [http://arxiv.org/abs/1606.03657](http://arxiv.org/abs/1606.03657).

## MNIST Latent Codes

- c<sub>1</sub> ~ Cat(K = 10, p = 0.1)
- c<sub>2</sub> ~ Unif(-1, 1)
- c<sub>3</sub> ~ Unif(-1, 1)

10 categorical and 2 continuous codes are sampled and concatenated with 62 noise variables z and then fed into a generator. Following are sample images created by a generator trained over 100 epochs on the MNIST training dataset consisting of 60000 samples.

### c<sub>1</sub> - 10 Categorical

Each row shows 10 random images generated from the same one-hot categorical vector. K = 10 and hence such 10 rows, one for each categorical code. The network learn the latent representations of the 10 different digits with some exceptions such as the digit 1.

<img src="./samples/c1_categorical.png" width="512px"/>

### c<sub>2</sub> - Continuous, Interpolation

Interpolating over c<sub>2</sub> shows that the network learns to distinguish the width of the digit.

<img src="./samples/c2_continuous_0.png" width="512px"/>
<img src="./samples/c2_continuous_9.png" width="512px"/>

### c<sub>3</sub> - Continuous, Interpolation

Interpolating over c<sub>3</sub> shows that the network learns the orientations.

<img src="./samples/c3_continuous_0.png" width="512px"/>
<img src="./samples/c3_continuous_9.png" width="512px"/>

## Run

### Train

```bash
python train.py --out-generator-filename ./trained/generator.model --gpu 0
```
