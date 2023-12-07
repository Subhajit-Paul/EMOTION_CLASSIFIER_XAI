# EMOTION_CLASSIFIER_XAI

# The GoEmotions Dataset by Google

The GoEmotions Dataset, curated by Google, offers several compelling reasons for its selection in natural language processing (NLP) tasks:
![Emotions](https://blogger.googleusercontent.com/img/a/AVvXsEigu_MmQ7zbqciHaEAl_rjZYNEPX6GyGEh9nkIoGOaMUg3BPCizBGJc-FhAMluHZHVX2cArth_0RgQVaEELUT3Y4oWv3V1h_ES5YjNxXJPre5YZy_2bG7ihLKjKOrQJTjEM-9SFLTFq6-Beo94ZS7yqslE-VFOH4xRlUX35rBVYtPskYGIv4DbBRiL08Q=s1213)

## 1. **Rich Emotion Annotations:**
   - GoEmotions provides a diverse and extensive set of emotion categories, allowing for a nuanced understanding of emotional expressions in text.
   - The dataset includes 28 emotion categories, providing a comprehensive range of emotions for training and evaluation.

## 2. **Fine-grained Emotion Labels:**
   - The dataset includes fine-grained emotion labels, allowing for a more detailed analysis of emotional nuances and distinctions within each category.
   - Fine-grained annotations contribute to the model's ability to capture subtle variations in emotional tone.

## 3. **Large-scale and Varied:**
   - GoEmotions is a large-scale dataset, featuring a substantial number of examples for model training.
   - The diversity of the dataset helps in building robust models that generalize well across various contexts and domains.

## 4. **Real-world Textual Data:**
   - The dataset is sourced from real-world textual data, making it representative of natural language found on online platforms.
   - Training models on real-world data enhances their performance in handling authentic and varied language expressions.

## 5. **Suitability for Transfer Learning:**
   - The dataset is well-suited for transfer learning tasks, enabling the pre-training of models on emotion-rich data before fine-tuning on specific downstream tasks.
   - T
## Find more at this [Blog](https://blog.research.google/2021/10/goemotions-dataset-for-fine-grained.html)

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

