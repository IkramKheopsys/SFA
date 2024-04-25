# User Guide

This Python script demonstrates the use of the `TextGeneration` and `get_embeddings` functions for text generation and obtaining embeddings from training data.

## Prerequisites

Before running the script, make sure you have Python installed on your system. You can download Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/).

## Installing Dependencies

To run the script, you'll need to install the required dependencies. You can do this by running the following command in your terminal:

``` pip install -r requirements.txt ```


## Using the Script

To use the text generation and embeddings functions, follow the steps below:

1. Ensure you have a file containing your training data in JSON format. The file should be in the following format:

   ```json
   {
       "sentence_1": "target_1",
       "sentence_2": "target_2",
       ...
   }
   
where each key is a training sentence and each value is the corresponding target



Place your training data file in the same directory as the main.py script.

Run the main.py script using the following command:
```python main.py```

This will execute the TextGeneration and get_embeddings functions with the specified training data file.




## Embedding Calculation

### Self-Attention

The embedding calculation in our model primarily relies on the self-attention mechanism. Here's how it works:

1. **Input Representation**: The input to the self-attention module is a batch of input sequences, where each sequence is represented as a collection of vectors. Each vector represents a token in the sequence. The dimensions of this input tensor are (batch_size x max_token_count x embed_size), where max_token_count represents the length of the longest sequence and embed_size represents the size of the position-encoded embedding vectors.

2. **Query, Key, Value Projection**: In the self-attention module, the input vectors are projected into query, key, and value vectors for each attention head. This is done using linear layers, with separate layers for each head. The number of attention heads determines the parallel and independent attention outputs calculated for each input sequence.

3. **Energy Computation**: Energy scores are computed for each pair of tokens in each input sequence. This is achieved by taking the dot product of the query vector of one token with the key vector of another token. The energy scores are computed independently for each attention head, resulting in a tensor of dimensions (head_count x batch_size x max_token_count x max_token_count).

4. **Masked Self-Attention**: To ensure that tokens cannot attend to themselves or to tokens that come after them in the sequence, a mask is applied to the energy scores. This mask zeros out the scores below the diagonal in the energy tensor.

5. **Attention Scores**: Softmax is applied to the energy scores along the last dimension to obtain attention scores for each token. These attention scores represent the importance of each token with respect to every other token in the sequence.

6. **Weighted Sum of Values**: Using the attention scores, a weighted sum of the value vectors is computed for each token. This weighted sum captures the context information from other tokens in the sequence, weighted by their attention scores.

7. **Output Combination**: The outputs from different attention heads are combined using a fully connected layer to produce the final output of the self-attention module, which has the same dimensions as the input tensor.

### Transformer Block

The embedding calculation also involves the Transformer block, which wraps around the self-attention module and adds additional layers. Here's how it contributes to the embedding calculation:

1. **Residual Connections**: The output of the self-attention module is added to the input embeddings, creating a residual connection. This allows the model to retain information from the original input while incorporating the context information learned by the self-attention module.

2. **Layer Normalization**: Normalization is applied to the output of the residual connection to stabilize training and improve performance.

3. **Feed-Forward Neural Network**: The output of the normalization layer is passed through a feed-forward neural network, consisting of linear layers with ReLU activation functions. This adds non-linearity to the embedding calculation and enables the model to learn complex patterns in the data.

### Transformer (Putting it all Together)

The embedding calculation is completed by stacking multiple Transformer blocks on top of each other. Each Transformer block takes the output of the previous block as input and applies the self-attention mechanism followed by additional layers to refine the embeddings. The final output of the Transformer module is a tensor representing the embeddings of the input sequences, with dimensions (batch_size x max_token_count x embed_size).

### Position Encoding

To incorporate positional information into the embeddings, position encoding is applied to the input embeddings before passing them through the Transformer blocks. Position encoding utilizes trigonometric functions to calculate a vector that encodes the position of each token in the sequence. This vector is added to the input embeddings to create position-encoded embeddings, ensuring that the model can differentiate between tokens based on their position in the sequence.
## How Inference Work
![image](https://github.com/IkramKheopsys/SFA/assets/113558455/e43621a9-8938-41e2-abfb-27aebef8e2af)

