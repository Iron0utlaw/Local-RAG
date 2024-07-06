# Local RAG For PDFs

## Setup

### Clone Repo

```
git clone https://github.com/Iron0utlaw/Local-RAG.git
```

### Create environment

```
python -m venv venv
```

### Activate environment

Linux/macOS:
```
source venv/bin/activate
```

Windows: 
```
.\venv\Scripts\activate
```

### Install requirements

```
pip install -r requirements.txt
```

Manual Torch Installation:

```
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Launch notebook

VS Code:

```
code .
```

Jupyter Notebook

```
jupyter notebook
```

## Guide
### Ask Function
```
ask(query, return_context)
```
- query -> Input to LLM
- return_context -> Boolean value, will return context as output if True

### Customization Aspects
``` python
llm_model.generate(**input_ids,
                                 temperature=0.7, # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                                 do_sample=True, # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                                 min_length=512, # minimum length of answer generated
                                max_new_tokens=1024) # how many new tokens to generate from prompt
```

``` python
base_prompt = """Based on the following context items, please answer the query.
                        Give yourself room to think by extracting relevant passages from the context before answering the query.
                        Don't return the thinking, only return the answer.
                        Make sure your answers are as explanatory as possible.
                        \nNow use the following context items to answer the user query:
                        {context}
                        \nRelevant passages: <extract relevant passages from the context here>
                        User query: {query}
                        Answer:"""
```

## What is it?

- Basically it is used for providing context to LLM which further utilizes it to generate output
- It came from Facebook
- R is Retrieval - Find relevant stuff according to the query, Retrieves the info from source provided
- A is Augmented - We augment the retrieved data with our input or prompt
- G is Generation - Generate data

## Why?

- Prevent hallucination
- More factual
- Works with custom data, LLMs trained on internet data provide a more general aspect of the response. Using custom data it can be more fine tuned and specific

## Points To Remember

- Pandas works well with list of dictionary  that is why we read pdf in such a way
- Embedding models and LLM don't work with INF token count
- We split data into sentences for easier to filter, can fit into embeddings model (384 tokens), more specificity to LLM
- Splitting into chunks gives us more granularity as chucks becomes smaller components making data  more specific
- Embedding can be thought of as catching the meaning in terms of numbers (dimension e.g. 768) → mpnet-base-v2
    - This gives the ability to search by meaning rather than keyword
    - To store vector or embeddings its better use a vector database, uses ANN to search
    - Embeddings can be used for any type of data
- Similarity Measures
    - Dot product
    - Cosine Similarity → When the direction measurement is important to computations e.g., Text Search
- Tokenizer is different from embedding models
- Flash Attention allows for faster tokenization process
- LLM output tokens, which have to be converted to text
- Temperature is important parameter, it basically controls how creative answers will be

## Major Steps

1. Get data from source, have as much as detail you will want use e.g., Page Number
2. Preprocessing of the raw data into chunks, sentences
3. Create embeddings
4. Semantic Search Implementation
5. Reranking of results of semantic search 

## Key terms

| Term | Description |
| ----- | ----- | 
| **Token** | A sub-word piece of text. For example, "hello, world!" could be split into ["hello", ",", "world", "!"]. A token can be a whole word,<br> part of a word or group of punctuation characters. 1 token ~= 4 characters in English, 100 tokens ~= 75 words.<br> Text gets broken into tokens before being passed to an LLM. |
| **Embedding** | A learned numerical representation of a piece of data. For example, a sentence of text could be represented by a vector with<br> 768 values. Similar pieces of text (in meaning) will ideally have similar values. |
| **Embedding model** | A model designed to accept input data and output a numerical representation. For example, a text embedding model may take in 384 <br>tokens of text and turn it into a vector of size 768. An embedding model can and often is different to an LLM model. |
| **Similarity search/vector search** | Similarity search/vector search aims to find two vectors which are close together in high-demensional space. For example, <br>two pieces of similar text passed through an embedding model should have a high similarity score, whereas two pieces of text about<br> different topics will have a lower similarity score. Common similarity score measures are dot product and cosine similarity. |
| **Large Language Model (LLM)** | A model which has been trained to numerically represent the patterns in text. A generative LLM will continue a sequence when given a sequence. <br>For example, given a sequence of the text "hello, world!", a genertive LLM may produce "we're going to build a RAG pipeline today!".<br> This generation will be highly dependant on the training data and prompt. |
| **LLM context window** | The number of tokens a LLM can accept as input. For example, as of March 2024, GPT-4 has a default context window of 32k tokens<br> (about 96 pages of text) but can go up to 128k if needed. A recent open-source LLM from Google, Gemma (March 2024) has a context<br> window of 8,192 tokens (about 24 pages of text). A higher context window means an LLM can accept more relevant information<br> to assist with a query. For example, in a RAG pipeline, if a model has a larger context window, it can accept more reference items<br> from the retrieval system to aid with its generation. |
| **Prompt** | A common term for describing the input to a generative LLM. The idea of "[prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering)" is to structure a text-based<br> (or potentially image-based as well) input to a generative LLM in a specific way so that the generated output is ideal. This technique is<br> possible because of a LLMs capacity for in-context learning, as in, it is able to use its representation of language to breakdown <br>the prompt and recognize what a suitable output may be (note: the output of LLMs is probable, so terms like "may output" are used). | 


**Note** This inspired by https://github.com/mrdbourke/simple-local-rag




