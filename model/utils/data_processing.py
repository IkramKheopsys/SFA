import torch
import json

class DataProcessor:
    def __init__(self, file_path):
        self.training_data = None
        self.file_path = file_path
        self.data_words = None
        self.target_words = None
        self.vocabulary_words = None
        self.word_to_ix = None
        self.ix_to_word = None


    def get_data_and_vocab(self):
        with open(self.file_path, 'r') as file:
            json_data = json.load(file)

        # Convert JSON data to Python list of dictionaries
        training_data_list = json_data['training_data']

        # Initialize an empty dictionary to store training data
        self.training_data = {}

        # Iterate over each entry in the list and add it to the training data dictionary
        for entry in training_data_list:
            self.training_data[entry['input']] = entry['output']

        self.data_words = [k for k, _ in self.training_data.items()]
        self.target_words = [v for _, v in self.training_data.items()]

        # Build vocabulary from training data
        self.vocabulary_words = list(set([element.lower() for nestedlist in [x.split(" ") for x in self.data_words] for element in nestedlist] + [element.lower() for nestedlist in [x.split(" ") for x in self.target_words] for element in nestedlist]))

        # Ensure <end> token is at the end of vocabulary list, and there's a blank at the beginning
        self.vocabulary_words.remove("<end>")
        self.vocabulary_words.append("<end>")
        self.vocabulary_words.insert(0, "")

        # Create mappings from word to index and index to word
        self.word_to_ix = {self.vocabulary_words[k].lower(): k for k in range(len(self.vocabulary_words))}
        self.ix_to_word = {v: k for k, v in self.word_to_ix.items()}
        return self.training_data, self.data_words, self.target_words, self.vocabulary_words, self.word_to_ix, self.ix_to_word


    def words_to_tensor(self, seq_batch, device=None):
        index_batch = []

        # Loop over sequences in the batch
        for seq in seq_batch:
            word_list = seq.lower().split(" ")
            indices = [self.word_to_ix[word] for word in word_list if word in self.word_to_ix]
            t = torch.tensor(indices)
            if device is not None:
                t = t.to(device)  # Transfer tensor to the specified device
            index_batch.append(t)

        # Pad tensors to have the same length
        return self.pad_tensors(index_batch)

    def tensor_to_words(self, tensor):
        index_batch = tensor.cpu().numpy().tolist()
        res = []
        for indices in index_batch:
            words = []
            for ix in indices:
                words.append(self.ix_to_word[ix].lower())  # Convert index to word
                if ix == self.word_to_ix["<end>"]:
                    break  # Stop when <end> token is encountered
            res.append(" ".join(words))
        return res

    def pad_tensors(self, list_of_tensors):
        tensor_count = len(list_of_tensors) if not torch.is_tensor(list_of_tensors) else list_of_tensors.shape[0]
        max_dim = max(t.shape[0] for t in list_of_tensors)  # Find the maximum length
        res = []
        for t in list_of_tensors:
            # Create a zero tensor of the desired shape
            res_t = torch.zeros(max_dim, *t.shape[1:]).type(t.dtype).to(t.device)
            res_t[:t.shape[0]] = t  # Copy the original tensor into the padded tensor
            res.append(res_t)

        # Concatenate tensors along a new dimension
        res = torch.cat(res)
        firstDim = len(list_of_tensors)
        secondDim = max_dim

        # Reshape the result to have the new dimension first
        return res.reshape(firstDim, secondDim, *res.shape[1:])
