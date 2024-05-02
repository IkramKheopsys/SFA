# -*- coding: utf-8 -*-
"""GPTFromScratchFlax.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/canyon289/GenAiGuidebook/blob/gh-pages/_sources/deepdive/GPTFromScratchFlax.ipynb

(gpt-from-scratch)=
# GPT from "scratch" in Flax

**TLDR**:
* This is the single notebook contains a full GPT that you can train at home  
  * It's a reimplementation of [Andrej Karpathy’s GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) from PyTorch into [Jax/Flax](why-flax).
* The things to pay attention to are
  * The tokenizer
  * The model structure and definition
  * How the model performance changes over various checkpoints

## The trained model and the initial model
Before we dive in here's the final model.
It was trained from the [Tiny Shakespare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

And here's where we started,
a bunch of random characters.

The rest of this notebook shows how we get from our random initial LLM to our trained one.

(why-flax)=
## Why Flax and Jax

Similar to PyTorch,
Flax and Jax are production grade tools that are used from individual research to massive multi cluster LLMs.

For this guide book there are some additional reasons.

* I find that the separation of state, model definition, and training loops in Flax make it easier to disambiguate the concepts.
* Being able to print the layer outputs as well helped with understanding and debugging
* Deterministic by default makes debugging simpler
* Now you have both a PyTorch and Flax/Jax reference!

This notebook dives straight into a Neural network model.
If you need a primer on Neural Networks start with [NN Quickstart](nn-quickstart)

## Additional Notes
Here's some additional details that will

### B, T, C Meaning
Karpathy uses the short hand the symbols B, T, C in his code and videos.
He explains the meaning here [here](https://youtu.be/kCc8FmEb1nY?t=1436)
which I'll write out for easy reference.

* **B - Batch** - How many training examples we’re using at once. Ths is just for computational efficiency
* **T - Time** - In a text input array, either encoded or decoded, the index that is being referenced
* **C - Channel** - The specific token "number", or index in the embedding array. This matches the

### Block Size is Context Window
The length of the input sequence the model gets for making the next character prediction

### Suggested Exercises
For additional hands on learning I’ve included two additional datasets in here,
along with all the code to do training and text generation

> abcdef
> A quick brown fox jumped over the lazy dog

These strings have specific properties.
The first has no repeating characters,
and it's easy to encode and decode to integers your head.
This means it's easy to get an overfit model with perfect predictions for debugging,
as well as inspect the parameters.
The second has no repeating alphabetic characters,
but it does have spaces adding just one additional layer of complexity.

</br>

For additional debugging and understanding of the you can perform an ablation study.'
That means removing parts of the model, making certain parameters smaller and larger, and seeing what happens.
Visualize parts of learned parameters using heatmaps and other tools to see what the model is learning.

```
qb = QuickBrownFox(block_size = block_size, batch_size=batch_size)
ab = Alphabet(block_size = block_size, batch_size=batch_size)
```

# Imports

!pip flax==0.7.5 tqdm matplotlib jaxlib optax
"""

seed = 1337
import jax
import jax.numpy as jnp

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

from tqdm.auto import tqdm
import logging
import requests

def create_logger():
    logger = logging.getLogger("notebook")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        consolehandler = logging.StreamHandler()
        consolehandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        consolehandler.setFormatter(formatter)
        logger.addHandler(consolehandler)

    return logger

logger = create_logger()

# Checking if Cuda is loaded
devices = jax.devices()
print(devices)

"""## Data Loaders

This text base class implements the tokenizer,
as well the string encoder and decoder.
This is not to be confused with the encoder and decoder portions of a Transformer which are differnt things.
"""

import abc
import tensorflow_datasets as tfds
class BaseTextProcesser:
    """Load text, with tokenizer encode and decode"""

    def __init__(self, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self._data = None

        self.text = self.set_text()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @abc.abstractmethod
    def set_text(self) -> str:
        """Sets the text corpus that going to be used"""
        return

    def encode(self, input_string):
        return [self.stoi[c] for c in input_string]

    def decode(self, token_iter):
        # Int for jax.ArrayImpl coercion
        return "".join([self.itos[int(c)] for c in token_iter])

    def batch_decoder(self, tokens_lists):
        return [self.decode(tokens) for tokens in tokens_lists]

    @property
    def data(self):
        if self._data is None:
            self._data = jnp.array(self.encode(self.text))
        return self._data

    def train_test_split(self, split=.9):
        """Return train test split of data.

        We'll do it the same as the tutorial without any fancy reshuffling or anything like that
        """

        # Let's now split up the data into train and validation sets
        n = int(split * len(self.data))  # first 90% will be training data, rest val
        train_data = self.data[:n]
        val_data = self.data[n:]
        return train_data, val_data

    def get_batch(self, key):
        """Depending on what's passed in it'll get batches

        Parameters
        ----------
        key: jax PRNG Key
        data: Jax array
        block_size int:
        batch_size int:

        Returns
        -------
        x: Training examples
        y: Target array
        ix: Indices
        """
        # Take batch size random samples of starting positions of text from Tiny Shakespeare
        # Jax require more arguments than pytorch for random sample
        ix = jax.random.randint(key=key, minval=0, maxval=len(self.data) - self.block_size,
                                shape=(self.batch_size,))

        # Each starting position of text take a snippet and stack all the text snippets together
        x = jnp.stack([self.data[i:i + self.block_size] for i in ix])

        # The training data is the same just one position offset
        y = jnp.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        return x, y, ix

"""Three datasets are included
* **Tiny Shakespeare** - This is the original from the tutorial
* **Alphabet** - Just the alphabet, no character repeats so the LLM should be able to memorize the next otoken
* **QuickBrownFox** - A sentence where letters don't repeat, but spaces do leading to a tiny bit of variability.

The second two datasets are included for debugging purposes.
One of the easiest ways to debug an LLM, or any Deep Learning model, is overfit a simple dataset.
If you model can't do that you have problems.
"""

class TinyShakespeare(BaseTextProcesser):

    def set_text(self):
        """Sets the text corpus that going to be used"""
        text = None
        if text is None:
            text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        return text

class QuickBrownFox(BaseTextProcesser):
    @abc.abstractmethod
    def set_text(self):
        """Sets the text corpus that going to be used"""
        text = "The quick brown fox jumped over the lazy dog"
        return text

class Alphabet(BaseTextProcesser):
    @abc.abstractmethod
    def set_text(self):
        """Sets the text corpus that going to be used"""
        text = "abcdefg"
        return text

"""## Random Keys
JAX, which Flax is based on,
by design requires explicit keys for its randomness.
This is counterintuitive, but is smart for so many reasons.
Read [the docs](https://jax.readthedocs.io/en/latest/jax.random.html) to learn more.
"""

root_key = jax.random.PRNGKey(seed=0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

"""## Model Constants"""

batch_size = 128 # How many independent sequences will we process in parallel?
block_size = 64 # What is the maximum context length for predictions?

"""## Load and Encode Data"""

ts = TinyShakespeare(block_size = block_size, batch_size=batch_size)
# ts = utils.QuickBrownFox(block_size = block_size, batch_size=batch_size)
# ts = utils.Alphabet(block_size = block_size, batch_size=batch_size)

xb, yb, ix = ts.get_batch(main_key)
ts.text[:50]

batch_size

ts.vocab_size

xb[:2]

len(ts.batch_decoder(xb))

"""## Model Parameters"""

# Model Parameters
vocab_size = ts.vocab_size
n_embd = 120
n_head = 6
n_layer = 6
dropout_rate = .4

# n_embd = 300
# n_head = 6
# n_layer = 6
#dropout = 0.2

global_mask = nn.make_causal_mask(xb)

global_mask.shape

"""### Multi Head Attention Head
Now this is not from scratch but I chose to use the precanned version to show what a production implementation would look like.
Andrej already does a great job explaining the internals of attention so I didn't feel repeating here would add any extra value at the moment.
"""

class MultiHeadAttention(nn.Module):
    """Combine single Attention Head into one here"""
    num_heads: int
    n_embd: int

    @nn.compact
    def __call__(self, x, training):
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=dropout_rate,
            deterministic=not training
        )(x, mask=global_mask)
        x = nn.Dense(self.n_embd)(x)
        return x

"""### Test Parameter Initialization"""

mha = MultiHeadAttention(
    # head_size=head_size,
    num_heads=n_head,
    n_embd=n_embd,
    )

input_x = jnp.ones((batch_size, block_size, n_embd))
logger.debug(f"{input_x.shape}")

params = mha.init({'params':params_key, "dropout": dropout_key}, input_x, training=True)

print(mha.apply(params, input_x, training=False).shape)

input_x = jnp.ones((batch_size, block_size, n_embd))
params = mha.init(root_key, input_x, training=False)

mha.apply(params, input_x, training=False);

"""## Feedforward Layer
(batch_size, block_size) -> (batch_size, block_size, n_embd)
"""

class FeedForward(nn.Module):

    @nn.compact
    def __call__(self, x, training):
        x = nn.Sequential([
            nn.Dense(4 * n_embd),
            nn.relu,
            nn.Dense(n_embd),
            nn.Dropout(dropout_rate, deterministic = not training)
        ])(x)
        return x

ff = FeedForward()

input_x = jnp.ones((batch_size, block_size, n_embd))
logger.debug(f"{input_x.shape}")

params = ff.init({'params':params_key, "dropout": dropout_key}, input_x, training=True)

print(ff.apply(params, input_x, training=False).shape)

print(ff.tabulate(root_key, input_x, training=False,
      console_kwargs={'force_terminal': False, 'force_jupyter': True}))

"""## Block"""

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, training):

        sa = MultiHeadAttention(
                            n_embd=n_embd,
                            num_heads=n_head,
                            )
        ff = FeedForward()

        x = x + sa(nn.LayerNorm(n_embd)(x), training=training)
        x = x + ff(nn.LayerNorm(n_embd)(x), training=training)
        return dict(x=x, training=training)

block = Block()

input_x = jnp.ones((batch_size, block_size, n_embd))
logger.debug(f"{input_x.shape=}")

block_params = block.init({'params':params_key, "dropout": dropout_key}, input_x, training=True)
logger.info(f"{block.apply(block_params, input_x, training=False)['x'].shape=}")

print(block.tabulate(root_key, input_x, training=False,
      console_kwargs={'force_terminal': True, 'force_jupyter': True},
                     column_kwargs = {'width': 400}))

"""### Full Model"""

vocab_size = ts.vocab_size

class BigramLanguageModel(nn.Module):

    @nn.compact
    def __call__(self, idx, training):
        logger.debug(f"In call {idx.shape=}")
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = nn.Embed(vocab_size, n_embd, name="TokenEmbedding")(idx) # (B,T,C)
        pos_emb = nn.Embed(block_size, n_embd, name="Position Embedding")(jnp.arange(T)) # (T,C)

        x = tok_emb + pos_emb # (B,T,C)
        x = nn.Sequential([Block() for _ in range(int(n_layer))])(x, training=training)["x"] # (B,T,C)
        x = nn.LayerNorm(n_embd, name="LayerNorm")(x) # (B,T,C)
        logits = nn.Dense(vocab_size, name="Final Dense")(x) # (B,T,vocab_size)
        return logits

    def generate(self, max_new_tokens):
        idx = jnp.zeros((1, block_size), dtype=jnp.int32)*4

        # We need to get this to enable correct random behavior later
        key = jax.random.PRNGKey(0)


        for i in range(max_new_tokens):
            logger.debug(f"In generate {i=}")

            # Get the predictions
            logger.debug(f"In generate {idx=}==========")
            logits = self.__call__(idx[:, -block_size:], training=False)
            logger.debug(f"In generate {logits.size=}")


            ## Focus only on the logits last time step
            # logits_last_t = logits[:, -1, :] # becomes (T)
            logits_last_t = logits[0, -1]

            # Due to the way randomness works in jax we have to generate subkeys to get new values
            # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng
            _, subkey = jax.random.split(key)

            # Update: Jax categorical wants unnormalized logprobabilities so we don't need to softmax
            # https://jax.readthedocs.io/en/latest/_autosummary/jax.random.categorical.html
            # sample from the distribution.
            idx_next = jax.random.categorical(subkey, logits_last_t)

            # Rotate the key
            key = subkey

            # append sampled index to the running sequence
            idx = jnp.atleast_2d(jnp.append(idx, idx_next))
            logger.debug(f"In generate after append {idx=}")

        return idx

bglm = BigramLanguageModel()

logger.setLevel(logging.DEBUG)

bglm = BigramLanguageModel()

input_x = jnp.ones((batch_size, block_size), dtype=jnp.int16)
logger.debug(f"{input_x.shape=}")
initial_params = bglm.init({'params':params_key, "dropout": dropout_key}, input_x, training=True)

initial_params["params"].keys()

print(bglm.tabulate({'params':params_key, "dropout": dropout_key}, input_x, training=False,
      console_kwargs={'force_terminal': False, 'force_jupyter': True},
                     column_kwargs = {'width': 400}))

"""### Sample Generation
This is with initial parameters that are totally random.
"""

logger.propagate = False
bglm = BigramLanguageModel()

idx = bglm.apply(initial_params, 50, method='generate')
ts.decode(idx.tolist()[0])

"""## Training"""

learning_rate = 1e-2

class TrainState(train_state.TrainState):
  key: jax.Array

state = TrainState.create(
  apply_fn=bglm.apply,
  params=initial_params,
  key=dropout_key,
  tx=optax.adam(learning_rate)
)

@jax.jit

def train_step(state: TrainState, inputs, labels):
  dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

  def cross_entropy_loss(params):
        logits = bglm.apply(params,
                            inputs,
                            training=True,
                            rngs={'dropout': dropout_key})
        logger.debug(logits.shape)

        # We use with integer labels method so we don't have to bother with one hot encoding
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))
        return loss, logits

  grad_fn = jax.value_and_grad(cross_entropy_loss, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

state.params["params"].keys()

state.step

"""## Training Loop
This model was trained on my RTX 4090.
It took about 30 minutes to train.
"""

logger.setLevel(logging.DEBUG)

eval_interval = 100
CKPT_DIR = 'ckpts'
epochs = 1000
_loss = []


import os

CKPT_DIR = os.path.abspath('ckpts/nn_compact0.orbax-checkpoint-tmp-1714402802417678')

train_key, dropout_key = jax.random.split(key=root_key, num=2)

for epoch in tqdm(range(epochs)):

    # Generate a new random key
    train_key = jax.random.fold_in(key=root_key, data=state.step)

    # Get a new batch
    xb, yb, ix = ts.get_batch(train_key)

    # Calculate the gradient
    state, loss = train_step(state, xb, yb)
    _loss.append(loss)

    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        print(f"step {epoch}: train loss {loss:.4f}")

    # # Update the model parameters
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR,
                               target=state,
                               keep_every_n_steps=100,
                               prefix='nn_compact',
                               overwrite=True,
                               step=epoch)

"""### Training Loss"""

fig, ax = plt.subplots()
ax.plot(np.arange(epochs), _loss)

ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss")

"""### Final Results
We can now use our weights to generate text.
At a glance it's not bad.
"""

idx.tolist()[0]

(ts.decode(idx.tolist()[0]))



logger.setLevel(logging.INFO)

idx = bglm.apply(state.params, 600, method='generate')
generation = ts.decode(idx.tolist()[0])
print(generation.strip())

logger.setLevel(logging.INFO)

idx = bglm.apply(state.params, 600, method='generate')
generation = ts.decode(idx.tolist()[0])
print(generation.strip())

"""### Comparing Checkpoints
What's interesting is going back in time and comparing generations from different checkpoints.
Since we saved earlier checkpoints we can load them and see what the LLM had learned up until that point.

#### Checkpoint 501
"""

ckpt_301 = checkpoints.restore_checkpoint(ckpt_dir=f'{CKPT_DIR}/nn_compact301', target=None)

idx = bglm.apply(ckpt_301["params"], 600, method='generate')
generation = ts.decode(idx.tolist()[0])
print(generation.strip())

"""#### Checkpoint 1101"""

ckpt_1101 = checkpoints.restore_checkpoint(ckpt_dir=f'{CKPT_DIR}/nn_compact1101', target=None)

idx = bglm.apply(ckpt_1101["params"], 600, method='generate')
generation = ts.decode(idx.tolist()[0])
print(generation.strip())

"""### Checkpoint 5901"""

ckpt_5901 = checkpoints.restore_checkpoint(ckpt_dir=f'{CKPT_DIR}/nn_compact5901', target=None)

idx = bglm.apply(ckpt_5901["params"], 600, method='generate')
generation = ts.decode(idx.tolist()[0])
print(generation.strip())