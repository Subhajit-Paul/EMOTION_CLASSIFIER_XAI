# EMOTION_CLASSIFIER_XAI
---
A simple Transformer based NLP algorithm that uses `BERT` pretrained parameters and the power of Explainable AI to understand the classification of Emotions on a given input. To enhance the interpretability of the model's decisions, saliency maps are employed. Saliency maps highlight the most influential words or tokens in a given input text that contribute to the model's prediction. By visualizing these salient regions, users can gain insights into which words or phrases the model relies on to make accurate emotion predictions.
---
## Deployed app will be launched soon

## Setup
### 1. Install necessary libraries from the `requirements.txt` using `pip install -r requirements.txt`
### 2. Go through the Preprocessing and Data Cleaning steps at `Data_Cleaning.ipynb`
### 3. Go through the Model Training Process at `training.ipynb`. More about the architecture see `bert_model_graph.pdf`
### 4. Go through the Model Inference and Explainability at `inference_and_explainability.ipynb`

## find the datasets here `https://drive.google.com/drive/folders/1-TCcCU2tAy-MJP_Pp8wSbJWCu6sJC2hP?usp=sharing` 
## find the model here `https://drive.google.com/drive/folders/1waz4OOSk86veZVFcEUMHfV7oFJSlFFHt?usp=sharing`

## To Get more Hands-On run app.py using `streamlit run app.py`

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
     
### Find more at this [Blog](https://blog.research.google/2021/10/goemotions-dataset-for-fine-grained.html)

# Overview of Preprocessing
The preprocessing steps are applied to the GoEmotions dataset, a manually annotated dataset designed for fine-grained emotion prediction, with a focus on emotions expressed in Reddit comments. The dataset encompasses 28 different emotions.

## 1. Loading Important Libraries
   - Various libraries are imported, such as `Pandas` for data manipulation, `NumPy` for numerical operations, and `NLTK` for text cleaning and processing.

## 2. Loading the Data
   - The dataset is loaded using the `HuggingFace` `Datasets` library, and it is split into training, testing, and validation sets.

## 3. Finding and Handling Out-of-Vocabulary Words
   - Out-of-vocabulary (OOV) words are identified by checking the coverage of the vocabulary in pre-trained word embeddings (GloVe). Preprocessing steps include handling contractions, wrong-spelled words, and emojis.

## 4. Saving Processed Splits as Texts and Labels
   - The preprocessed text and labels for the training, testing, and validation sets are saved as compressed numpy arrays for future use.

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

