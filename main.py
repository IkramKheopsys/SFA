from model.test_models.functions import text_generation,get_embeddings


def main():

    file_path = 'training_data.json'

    model_output = text_generation(file_path)

    print("\n\n")
    print("Text Generation:")
    print(model_output)

    Embedding_output = get_embeddings(file_path)
    print("\n\n")
    print("Embeddings:")
    print(Embedding_output)




if __name__ == "__main__":
    main()
