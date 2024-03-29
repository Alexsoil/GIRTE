import random
import torch
import os
import sys
import gc
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
            input_data.append(document_words)
    
    # print input_data
    print('Data loaded.')

    encoding = tokenizer.batch_encode_plus(
        input_data,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True,
        is_split_into_words=True
    )
    print('Encoding complete.')

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    # print(f'Input Id: {input_ids}')
    # print(f'Attention mask: {attention_mask}')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state

    print('Embeddings created.')
    print(word_embeddings.shape)
    
    tokenized_text = []

    # for id in input_ids:
    #     decoded_text = tokenizer.decode(id, skip_special_tokens=True)
    #     print(f'Decoded text: {decoded_text}')
    #     tokenized_text.append(tokenizer.tokenize(decoded_text))
    #     print(f'Tokenized text: {tokenized_text}')

    for doc_embedding in word_embeddings:
        for word_embedding in doc_embedding[0:2]:
            print(f'Tensor: {word_embedding[0:10]}')
            print('\n')

    
    
    print('Done')

except Exception as err:
    print(f'Door stuck: {err}')