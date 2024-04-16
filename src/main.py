import random
import torch
import os
import sys
import gc
import networkx as nx
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


try:
    # Seed for multithreading
    random_seed = 69
    random.seed(random_seed)

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Loading pretrained Bert Models
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    data_location = os.path.join('collections', sys.argv[1], 'docs')

    # For each document...
    for filename in os.listdir(data_location):
        with open(os.path.join(data_location, filename), 'r') as file:
            document = file.read()
            document_words = document.split('\n')
            print(document_words)
            # print input_data
            print('Data loaded.')

            # Encode document words into tokens
            encoding = tokenizer.__call__(
                document_words,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
                is_split_into_words=True
            )

            print('Encoding complete.')

            input_ids = encoding['input_ids']
            print(f'Input ID: {input_ids}')
            attention_mask = encoding['attention_mask']
            print(f'Attention mask: {attention_mask}')

            # Get last layer of BERT as tensors
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                word_embeddings = outputs.last_hidden_state

            print('Embeddings created.')
            decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f'Decoded Text: {decoded_text}')
            tokenized_text = tokenizer.tokenize(decoded_text)
            print(f'Tokenized Text: {tokenized_text}')
            encoded_text = tokenizer.encode(' '.join(document_words), return_tensors='pt')
            print(f'Encoded Text: {encoded_text}')
            print(word_embeddings.shape)
            
            # Create graph for this document
            G = nx.Graph()

            # Choose the next word
            for tok, i in zip(tokenized_text, word_embeddings[0]):
                
                # if not in the graph, hold it
                if G.has_node(tok) is False:
                    # iterate through the rest of the words
                    embedding_combiner = []
                    for tok_comp, j in zip(tokenized_text, word_embeddings[0]):
                        # if any word is equal to the one held, append them to a list
                        if tok_comp == tok:
                            # print('match')
                            embedding_combiner.append(j)
                    # when every word has been checked, take the mean of the embeddings for held word
                    stacked_embedding = torch.stack(embedding_combiner)
                    # Shape: [X, 768]
                    mean_embedding = torch.mean(stacked_embedding, dim=0)
                    # Shape: [768]
                    G.add_node(tok, tensor=mean_embedding)
                else:
                    # if it already is in the graph, skip it
                    continue
    
                ### TODO: Cosine similarity - Theta - Edge creation
    
            # Print graph nodes and first tensor element
            for node in G.nodes(data='tensor'):
                print(f'{node[0]} -> {node[1][0]}')
            break    
    
    print('Done')

except Exception as err:
    print(f'Door stuck: {err}')