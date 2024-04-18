import random
import torch
import os
import sys
import pickle
import time
import networkx as nx
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


try:
    # Total timer start
    total_t_start = time.time()
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
    model_location = os.path.join('picklejar', sys.argv[1])
    max_iterations = int(sys.argv[3])
    if __debug__:
        max_iterations = 1

    # For each document...
    iterations = 0
    for filename in tqdm(sorted(os.listdir(data_location))):
        with open(os.path.join(data_location, filename), 'r') as file:
            # Document timer start
            doc_t_start = time.time()
            document = file.read()
            document_words = document.split('\n')
            if __debug__:
                print(document_words)
                print('Data loaded.')
            # Document timer end
            doc_t_end = time.time()

            # Encoding + Embedding timer start
            bert_t_start = time.time()
            # Encode document words into tokens
            encoding = tokenizer.__call__(
                document_words,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
                is_split_into_words=True
            )
            if __debug__:
                print('Encoding complete.')

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            if __debug__:
                print(f'Input ID: {input_ids}')
                print(f'Attention mask: {attention_mask}')

            # Get last layer of BERT as tensors
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                word_embeddings = outputs.last_hidden_state
            if __debug__:
                print('Embeddings created.')
            decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            tokenized_text = tokenizer.tokenize(decoded_text)
            encoded_text = tokenizer.encode(' '.join(document_words), return_tensors='pt')
            if __debug__:
                print(f'Decoded Text: {decoded_text}')
                print(f'Tokenized Text: {tokenized_text}')
                print(f'Encoded Text: {encoded_text}')
                print(word_embeddings.shape)
            # Encoding + Embedding timer end
            bert_t_end = time.time()

            # Graph time start
            graph_t_start = time.time()
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
                    # Shape: [768]w
                    G.add_node(tok, tensor=mean_embedding)
                else:
                    # if it already is in the graph, skip it
                    continue
    
                ### TODO: Cosine similarity - Theta - Edge creation
    
            for node_outer in G.nodes(data='tensor'):
                for node_inner in G.nodes(data='tensor'):
                    similarity = cosine_similarity(torch.reshape(node_outer[1], (1, -1)), torch.reshape(node_inner[1], (1, -1)))
                    # sys.argv[2] = theta
                    if similarity[0][0] < 1 - float(sys.argv[2]):
                        G.add_edge(node_outer[0], node_inner[0], weight=similarity[0][0])
            # Graph timer end
            graph_t_end = time.time()

            if __debug__:
                # Print nodes
                print('Node -> 1st value of Tensor')
                for node in G.nodes(data='tensor'):
                    print(f'{node[0]} -> {node[1][0]}')

                # Print edges
                print('Edge -> Weight')
                for edge in G.edges(data='weight'):
                    print(f'{edge[0]} -> {edge[1]} ({edge[2]})')

            # Graph Information
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            print(f'Graph with {num_nodes} nodes, {num_edges}/{int((num_nodes * (num_nodes - 1)) / 2) + num_nodes} possible edges.')
            print(f'Iteration total time: {(graph_t_end - doc_t_start):.2f}\tDocument: {(doc_t_end - doc_t_start):.2f}\tEmbedding: {(bert_t_end - bert_t_start):.2f}\tGraph: {(graph_t_end - graph_t_start):.2f}')
            
            file.close()
        
        if not __debug__:
            # Saving as pickle data
            os.makedirs(os.path.join(model_location), exist_ok=True)
            with open(os.path.join(model_location, filename + '.graph'), 'wb') as picklefile:
                pickle.dump(G, picklefile)
                file.close()

        # with open(os.path.join(model_location, filename), 'rb') as pickfile:
        #     abnaroz = pickle.load(pickfile)
        # print(abnaroz.number_of_nodes())
        iterations += 1
        if iterations >= max_iterations:
            break
    # Total timer end
    total_t_end = time.time()
    print(f'Processed {iterations} documents.\tTotal time: {(total_t_end - total_t_start)/60:.2f} minutes')

except Exception as err:
    print(f'Door stuck: {err}')