
# Define Transformer block module
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_count):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, head_count)  # Self-attention layer
        self.norm1 = nn.LayerNorm(embed_size)  # Layer normalization
        self.norm2 = nn.LayerNorm(embed_size)  # Layer normalization

        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, embeddings):
        attention = self.attention(embeddings)

        # Apply residual connections and layer normalization
        out = self.norm1(attention + embeddings)
        out = attention + self.feed_forward(out)
        out = self.norm2(out)
        return out

# Define Transformer module
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, head_count):
        super(Transformer, self).__init__()
        self.embed_size = embed_size  # Size of word embeddings
        self.vocab_size = vocab_size  # Size of vocabulary
        self.word_embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer

        # List of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, head_count) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Final linear layer to produce logits

    def forward(self, input_tokens, mask=None):
        batch_size, token_count = input_tokens.shape[:2]
        out = self.word_embedding(input_tokens)  # Obtain word embeddings

        # Compute position encodings and add to word embeddings
        positions = torch.arange(0, token_count).expand(batch_size, token_count).to(input_tokens.device)
        position_encoding = self.position_encoding(positions, self.embed_size)
        out += position_encoding.reshape(out.shape)

        # Pass through each transformer block
        for layer in self.layers:
            out = layer(out)

        # Produce logits for the final token in each sequence
        out = self.fc_out(out[:, -1, :].reshape(batch_size, self.embed_size)).reshape(batch_size, self.vocab_size)
        return torch.nn.functional.softmax(out, dim=1)  # Apply softmax to obtain probabilities

    def position_encoding(self, positions, embed_size):
        # Compute position encoding for each position and dimension
        angle_rads = self.get_angles(
            positions.unsqueeze(2).float(),
            torch.arange(embed_size)[None, None, :].float().to(positions.device),
            embed_size
        )
        sines = torch.sin(angle_rads[:, :, 0::2])  # Compute sine of angle for even dimensions
        cosines = torch.cos(angle_rads[:, :, 1::2])  # Compute cosine of angle for odd dimensions
        pos_encoding = torch.cat([sines, cosines], dim=-1)  # Concatenate sine and cosine values
        pos_encoding = pos_encoding[None, ...]
        return pos_encoding

    def get_angles(self, pos, i, embed_size):
        # Compute angle rate for each position and dimension
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / embed_size)
        return pos * angle_rates

# Function to train the model recursively over each sequence and token
def train_recursive(model, data, targets, optimizer, criterion):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Zero the gradients
    total_loss = 0  # Initialize total loss
    batch_size, token_count, token_count_out = data.shape[0], data.shape[1], targets.shape[1]

    # Loop over sequences in the batch
    for b in range(batch_size):
        end_encountered = False
        cur_count = 0
        # Loop over tokens in the sequence
        while not end_encountered:
            target_vector = torch.zeros(model.vocab_size).to(data.device)  # Initialize target vector

            if cur_count != token_count_out:
                expected_next_token_idx = targets[b, cur_count]  # Get index of expected next token
                target_vector[expected_next_token_idx] = 1  # Set the corresponding element of the target vector to 1

            # Concatenate current input and output tokens and pass through model
            if cur_count > 0:
                model_input = data[b].reshape(token_count).to(data.device)
                part_of_output = targets[b, :cur_count].to(data.device)
                model_input = torch.cat((model_input, part_of_output))
            else:
                model_input = data[b]
            out = model(model_input.reshape(1, token_count + cur_count))

            # Compute loss and accumulate total loss
            loss = criterion(out, target_vector.reshape(out.shape))
            total_loss += loss
            cur_count += 1

            # Stop when the end of the sequence is reached
            if cur_count > token_count_out:
                end_encountered = True

    # Backpropagate gradients and update model parameters
    total_loss.backward()
    optimizer.step()
    return total_loss.item() / batch_size

# Function to perform inference recursively for each sequence in a batch
def infer_recursive(model, input_vectors, max_output_token_count=10):
    model.eval()  # Set model to evaluation mode
    outputs = []

    # Loop over sequences in the batch
    for i in range(input_vectors.shape[0]):
        print(f"Infering sequence {i}")
        input_vector = input_vectors[i].reshape(1, input_vectors.shape[1])
        predicted_sequence = []
        wc = 0  # Initialize word count

        with torch.no_grad():  # Disable gradient computation
            while True:
                output = model(input_vector)  # Pass current input through model
                predicted_index = output[0, :].argmax().item()  # Get index of predicted token
                predicted_sequence.append(predicted_index)  # Append predicted index to sequence
                # Stop when <end> token is predicted or the maximum output length is reached
                if predicted_index == word_to_ix['<end>'] or wc > max_output_token_count:
                    break
                # Append predicted token to input and increment word count
                input_vector = torch.cat([input_vector, torch.tensor([[predicted_index]])], dim=1)
                wc += 1
        outputs.append(torch.tensor(predicted_sequence))  # Append predicted sequence to outputs
    outputs = pad_tensors(outputs)  # Pad predicted sequences to the same length
    return outputs

