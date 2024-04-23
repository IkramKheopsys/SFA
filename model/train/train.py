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

