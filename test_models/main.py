# Function to demonstrate training and inference
import torch
import torch.nn as nn
import torch.optim as optim
import pprint

# Assume these functions and classes are available in your package
from packaging.models import Transformer
from packaging.utils import words_to_tensor, tensor_to_words, train_recursive, infer_recursive

def example_training_and_inference():
    # Get model hyperparameters from vocabulary size
    vocab_size = len(word_to_ix)
    embed_size = 512
    num_layers = 4
    heads = 3

    # Create model, optimizer, and loss function
    device = torch.device("cpu")
    model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    # Convert training data to tensors
    data = words_to_tensor(data_words, device=device)
    targets = words_to_tensor(target_words, device=device)

    # Train model for 55 epochs
    for epoch in range(55):
        avg_loss = train_recursive(model, data, targets, optimizer, criterion)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    # Perform inference on training data
    input_vector = words_to_tensor(data_words, device=device)
    predicted_vector = infer_recursive(model, input_vector)
    predicted_words = tensor_to_words(predicted_vector)

    # Print training data and model output
    print("\n\n\n")
    print("Training Data:")
    pprint.pprint(training_data)
    print("\n\n")
    print("Model Inference:")
    result_data = {data_words[k]: predicted_words[k] for k in range(len(predicted_words))}
    pprint.pprint(result_data)
