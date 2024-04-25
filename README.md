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

In colab you can clone this repo, go to the SFA file :
```import os
os.chdir('SFA')
!pwd
```
And Run the main.py script using the following command:

```!python main.py```
## Dataset Format
The dataset used for training should follow a specific format, as exemplified below:

```
{
    "training_data": [
        {
            "input": "what is your name",
            "output": "my name is abdoul <end>"
        },
        {
            "input": "what's your company name",
            "output": "My company's name is kheopsys <end>"
        },
        {
            "input": "who is Abdoul",
            "output": "Abdoul is the ceo of kheopsys <end>"
        },
        {
            "input": "where is Kheopsys",
            "output": "It's located in Paris <end>"
        },

        {
            "input": "who are you",
            "output": "mini gpt model <end>"
        }
    ]
}

```

Each training example consists of an "input" and an "output" field, representing the input sequence and its corresponding target sequence, respectively. The "<end>" token signifies the end of the output sequence.

### Additional Datasets
Here are some other datasets that may be relevant for our purposes:

First Dataset (45 rows): https://huggingface.co/datasets/LLMao/standard_qa/blob/main/data/train-00000-of-00001.parquet

Second Dataset (2.56k rows): https://huggingface.co/datasets/Anthropic/llm_global_opinions

Third Dataset (Bollywood, 6k rows): https://huggingface.co/datasets/LLMao/qa_Bollywood?row=32

Fourth Dataset (Short Answers, 36k rows): https://huggingface.co/datasets/UCLNLP/adversarial_qa?row=12

These datasets can be processed similarly to the provided example, where each answer is concatenated with "<end>" to signify the end of the sequence. Adjustments may be necessary based on the specific dataset being used.


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

## Training Process

The train_recursive function is utilized to train the model on input data and target data. It applies Transformer blocks to each token in the sequence in parallel, allowing the model to capture long-range dependencies efficiently. The self-attention mechanism enables the model to consider interactions between tokens, while residual connections and layer normalization stabilize training and improve performance. The function calculates the loss between the model's predictions and the target data, backpropagates the gradients, and updates the model parameters using an optimizer.



## How Inference Work
The inference unction recursively generates output sequences for each input sequence using the trained model. It predicts the next token/word iteratively based on the model's output probabilities, updating the input tensor for each prediction until a stopping criterion is met. Finally, it returns the predicted sequences as a padded tensor for further processing or evaluation
Here is a diagram with a different example to demonstrate how inference works –


![image](https://github.com/IkramKheopsys/SFA/assets/113558455/e43621a9-8938-41e2-abfb-27aebef8e2af)

