import streamlit as st
import torch
import transformers
from transformers import BertTokenizer, BertModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# parameters
MAX_LEN = 50
TRAIN_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
emotions_dict = {
    0: {'emotion': 'admiration', 'emoji': '😊'},
    1: {'emotion': 'amusement', 'emoji': '😄'},
    2: {'emotion': 'anger', 'emoji': '😠'},
    3: {'emotion': 'annoyance', 'emoji': '😒'},
    4: {'emotion': 'approval', 'emoji': '👍'},
    5: {'emotion': 'caring', 'emoji': '❤️'},
    6: {'emotion': 'confusion', 'emoji': '😕'},
    7: {'emotion': 'curiosity', 'emoji': '🤔'},
    8: {'emotion': 'desire', 'emoji': '😍'},
    9: {'emotion': 'disappointment', 'emoji': '😞'},
    10: {'emotion': 'disapproval', 'emoji': '👎'},
    11: {'emotion': 'disgust', 'emoji': '🤢'},
    12: {'emotion': 'embarrassment', 'emoji': '😳'},
    13: {'emotion': 'excitement', 'emoji': '😃'},
    14: {'emotion': 'fear', 'emoji': '😨'},
    15: {'emotion': 'gratitude', 'emoji': '🙏'},
    16: {'emotion': 'grief', 'emoji': '😢'},
    17: {'emotion': 'joy', 'emoji': '😃'},
    18: {'emotion': 'love', 'emoji': '❤️'},
    19: {'emotion': 'nervousness', 'emoji': '😬'},
    20: {'emotion': 'optimism', 'emoji': '😊'},
    21: {'emotion': 'pride', 'emoji': '🏆'},
    22: {'emotion': 'realization', 'emoji': '🤯'},
    23: {'emotion': 'relief', 'emoji': '😅'},
    24: {'emotion': 'remorse', 'emoji': '😔'},
    25: {'emotion': 'sadness', 'emoji': '😢'},
    26: {'emotion': 'surprise', 'emoji': '😲'},
    27: {'emotion': 'neutral', 'emoji': '🙂'}
}

# the finetunable model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 28)

    def forward(self, input_ids, mask, token_type_ids):
        _, output_1= self.l1(input_ids = input_ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
  
# defining custom dataset  
class CustomDataset(Dataset):
    def __init__(self, X, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = X
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

# instantiating the model
model = BERTClass()
model.to(device)
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))

# to get logits from the model 
def get_emotion(text):
    data = CustomDataset([text], tokenizer, MAX_LEN)
    infer = DataLoader(data)

    for _, data in enumerate(infer):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        with torch.inference_mode():
            outputs = model(ids, mask, token_type_ids)

    return outputs

# to get Saliency plot
def get_saliency_plot(input_text):
    tokens = tokenizer.encode_plus(input_text, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    token_type_ids = tokens['token_type_ids'].to(device)
    
    # Define the baseline (reference) for integrated gradients
    ref_token_id = tokenizer.pad_token_id  # Use pad token for baseline
    ref_input_ids = torch.tensor([[ref_token_id] * input_ids.shape[1]]).to(device)
    
    # Instantiate the LayerIntegratedGradients object
    lig = LayerIntegratedGradients(model, model.l1.embeddings)
    
    tar = range(28)
    attributions = [lig.attribute(inputs=(input_ids, attention_mask, token_type_ids),
                                        baselines=(ref_input_ids, torch.zeros_like(attention_mask), torch.zeros_like(token_type_ids)),
                                        target=i).squeeze(0).detach().cpu().numpy().sum(axis=1) for i in tar]
    
    ticky = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    # Plot the saliency map using a heatmap
    plt.figure(figsize=(17, 10))
    sns.heatmap(attributions, xticklabels=[tokenizer.decode(token_id).replace(" ", "") for token_id in tokens['input_ids'][0]], cmap="viridis", yticklabels=ticky)
    plt.title('Saliency Map')
    return plt

st.title("Emotion Classifier App")
st.markdown(
    "This app uses a deep learning model to predict the emotion conveyed in a sentence."
)

# User Input Section
input_text = st.text_input(label="Enter a sentence (max 50 words):")

# Input Validation
if len(input_text.split()) > 50:
    st.warning("Error: The sentence must contain fewer than 50 words.")

# Emotion Classification
if input_text:
    emotion_result = get_emotion(input_text)[0]
    predicted_emotion = emotions_dict[torch.argmax(emotion_result).item()]
    formatted_emotion = f"{predicted_emotion['emotion'].capitalize()} {predicted_emotion['emoji']}"
    st.subheader(f"Predicted Emotion {formatted_emotion}")

    # Saliency Plot
    st.markdown("### Emotion Saliency Plot")
    st.pyplot(get_saliency_plot(input_text).show())

    # Additional Information
    st.info(
        "The prediction is based on a transformer model trained on emotional text data. "
        "Note that the accuracy may vary depending on the complexity of the sentence."
    )

    # Credits and References
    st.markdown("#### Credits:")
    st.markdown(
        "This app uses a pre-trained emotion transformer model, finetuned on a 28 label Dataset"
        "For more details, refer to the [GitHub repository](https://github.com/Subhajit-Paul/EMOTION_CLASSIFIER_XAI/)."
    )
