import torch


def evaluation_metrics(predicted_vector, input_vector):
    num_correct_words = 0
    # Loop over sequences in the batch
    for i in range(predicted_vector.shape[0]):
        predicted_sequence = predicted_vector[i]  # Get predicted sequence
        input_sequence = input_vector[i]  # Get input sequence

        # Count the number of words correctly predicted
        for j in range(min(predicted_sequence.shape[0], input_sequence.shape[0])):
            if predicted_sequence[j] == input_sequence[j]:
                num_correct_words += 1
        score = + num_correct_words / len(input_sequence)

    return score / predicted_vector.shape[0]

