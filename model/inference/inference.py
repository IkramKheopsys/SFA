
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
