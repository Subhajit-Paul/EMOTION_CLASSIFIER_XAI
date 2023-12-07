# EMOTION_CLASSIFIER_XAI



# BERTClass Model Architecture

The `BERTClass` model is composed of three main components: `l1`, `l2`, and `l3`.

## `l1`: BertModel

The `BertModel` consists of the following sub-modules:

### BertEmbeddings

- `word_embeddings`: Embedding layer with 105,879 tokens, each represented by a vector of size 768.
- `position_embeddings`: Embedding layer with 512 positions, each represented by a vector of size 768.
- `token_type_embeddings`: Embedding layer with 2 token types, each represented by a vector of size 768.
- `LayerNorm`: Layer normalization with 768 features.
- `dropout`: Dropout layer with a dropout probability of 0.1.

### BertEncoder

- `layer`: ModuleList containing 12 instances of `BertLayer`.
  
  #### BertLayer

  - `attention`: `BertAttention` module with self-attention mechanism.
  
    ##### BertSelfAttention

    - `query`: Linear layer mapping 768 input features to 768 output features.
    - `key`: Linear layer mapping 768 input features to 768 output features.
    - `value`: Linear layer mapping 768 input features to 768 output features.
    - `dropout`: Dropout layer with a dropout probability of 0.1.

    ##### BertSelfOutput

    - `dense`: Linear layer mapping 768 input features to 768 output features.
    - `LayerNorm`: Layer normalization with 768 features.
    - `dropout`: Dropout layer with a dropout probability of 0.1.

  - `intermediate`: `BertIntermediate` module.
  
    ##### BertIntermediate

    - `dense`: Linear layer mapping 768 input features to 3072 output features.
    - `intermediate_act_fn`: GELU activation function.

  - `output`: `BertOutput` module.
  
    ##### BertOutput

    - `dense`: Linear layer mapping 3072 input features to 768 output features.
    - `LayerNorm`: Layer normalization with 768 features.
    - `dropout`: Dropout layer with a dropout probability of 0.1.

### BertPooler

- `dense`: Linear layer mapping 768 input features to 768 output features.
- `activation`: Hyperbolic tangent (Tanh) activation function.

## `l2`: Dropout Layer

- Dropout layer with a dropout probability of 0.3.

## `l3`: Linear Layer

- Linear layer mapping 768 input features to 28 output features.

