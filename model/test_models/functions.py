import torch
import torch.nn as nn
import torch.optim as optim
from model.models.transformer import Transformer
from model.train.train import train_recursive
from model.inference.inference import infer_recursive
from model.utils.data_processing import DataProcessor
from model.evaluation.evaluation import evaluation_metrics

def text_generation(file_path):
    processor = DataProcessor(file_path)
    training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word = processor.get_data_and_vocab()
    vocab_size = len(word_to_ix)
    embed_size = 512
    num_layers = 4
    heads = 3
    device = torch.device("cpu")
    model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    data = processor.words_to_tensor(data_words, device=device)
    targets = processor.words_to_tensor(target_words, device=device)
    # Train model for 55 epochs
    for epoch in range(5):
        avg_loss = train_recursive(model, data, targets, optimizer, criterion)

    input_vector = processor.words_to_tensor(data_words, device=device)
    print("Input vector = ", input_vector)
    predicted_vector = infer_recursive(model, input_vector, word_to_ix, processor)
    predicted_words = processor.tensor_to_words(predicted_vector)
    result_data = {data_words[k]: predicted_words[k] for k in range(len(predicted_words))}
    evaluation = evaluation_metrics(predicted_vector,input_vector)
    print('evaluation metrics = :', evaluation)
    return result_data


def get_embeddings(file_path):
    processor = DataProcessor(file_path)
    training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word = processor.get_data_and_vocab()

    # Get model hyperparameters from vocabulary size
    vocab_size = len(word_to_ix)
    embed_size = 512
    num_layers = 4
    heads = 3

    device = torch.device("cpu")
    model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)

    data = processor.words_to_tensor(data_words, device=device)

    model.eval()
    with torch.no_grad():
        embeddings = model(data,from_get_embeddings=True)

    return embeddings
