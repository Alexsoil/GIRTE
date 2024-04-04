import random
import torch
import os
import sys
import gc
import networkx as nx
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


try:
    random_seed = 69
    random.seed(random_seed)

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    data_location = os.path.join('collections', sys.argv[1], 'docs')

    input_data = []

    for filename in os.listdir(data_location):
        with open(os.path.join(data_location, filename), 'r') as file:
            document = file.read()
            document_words = document.split('\n')
            print(document_words)
            # print input_data
            print('Data loaded.')

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
            # G.add_nodes_from(input_ids[0][1:-1])
            # for node in G.nodes:
            #     print(tokenizer.decode(node))
            for tok, i in zip(tokenized_text, word_embeddings[0]):
                # print(tok)
                for tok_comp, j in zip(tokenized_text, word_embeddings[0]):
                    if G.has_node(tok_comp) is False:
                        G.add_node(tok_comp)
                    cos = cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0]
                    # G.add_edge(i,j, weight=cos)
                    # print(f'{tok}/{tok_comp} -> {cos}')
            print(G.nodes)
            break    
    
    print('Done')

except Exception as err:
    print(f'Door stuck: {err}')