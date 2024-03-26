# GIRTE
Graph-Based Information Retrieval with Transformer Embeddings

## Current Goal:
- Load data to the system
- Use BERT to transform into embeddings
- Create Graph for each document (embeddings = Vertices, similarity = Edges)

## TODO list

Combine Graph Methods with Transformer Models

Very abstract:
Convert all data into transformer embeddings
For each embedding, make a node v in G(V,E)
Create similarity metric (?)
If two embeddings v1, v2 have similarity above a threshold (?), create edge e = (v1, v2)
Result: Graph of Embeddings and Similarity between them

For Retrieval:
Concert terms of query into embeddings
Cosine similarity between query term and document term embeddings
Calculate total document similarity to query and rank

## Questions to consider
- What kind of data are we working with? -> Data in collections folder
- Is it static or dynamic? -> It shouldn't matter: If we get new documents, we just fine-tune
- Which parts/functions can we use modules/libraries for? -> Anything we can find
- What must be customly implemented? -> Whatever has to

## Variables to consider
- How the transformer is initialized 
- How the similarity metric is calculated
- What's the threshold for creating an edge
