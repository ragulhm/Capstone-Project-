import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np

# ------------------------------------------------
# Page Setup
# ------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Bot Detection",
    layout="wide",
    page_icon="🤖"
)

st.title("🧠 Explainable AI Bot Detection Dashboard")

device = torch.device("cpu")

# ------------------------------------------------
# Preprocessing
# ------------------------------------------------
def preprocess_text(text):

    steps = []
    steps.append(("Original", text))

    text = re.sub(r"http\S+", "", text)
    steps.append(("URL Removed", text))

    text = re.sub(r"<.*?>", "", text)
    steps.append(("HTML Removed", text))

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    steps.append(("Special Characters Removed", text))

    text = text.lower().strip()
    steps.append(("Lowercase", text))

    return steps, text

# ------------------------------------------------
# Model Architecture
# ------------------------------------------------
class BotDetectionModel(nn.Module):

    def __init__(self, model_name):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_attentions=True
        )

        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self,input_ids,attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0,:]

        prob = self.classifier(cls)

        return prob, outputs

# ------------------------------------------------
# Models
# ------------------------------------------------
MODELS = {
    "BERT":("bert-base-uncased","bert_base_10epoch.pth"),
    "RoBERTa":("roberta-base","roberta_10epoch.pth"),
    "DistilBERT":("distilbert-base-uncased","distilbert_10epoch.pth"),
    "XLM-RoBERTa":("xlm-roberta-base","xlm_roberta_10epoch.pth")
}

# ------------------------------------------------
# Model Loader
# ------------------------------------------------
@st.cache_resource
def load_model(model_name, weight_path):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = BotDetectionModel(model_name)

    model.load_state_dict(
        torch.load(weight_path,map_location=device)
    )

    model.eval()

    return model, tokenizer

# ------------------------------------------------
# Input
# ------------------------------------------------
tweet = st.text_area("Enter Tweet", height=120)

# ------------------------------------------------
# Run
# ------------------------------------------------
if st.button("Run Explainable AI"):

    if tweet.strip()=="":
        st.warning("Enter a tweet")
        st.stop()

# ------------------------------------------------
# Pipeline
# ------------------------------------------------
    st.header("⚙️ AI Processing Pipeline")

    st.markdown("""
User Tweet  
⬇  
Preprocessing  
⬇  
Tokenization  
⬇  
Token IDs  
⬇  
Transformer Encoder  
⬇  
Hidden States  
⬇  
Neural Network  
⬇  
Bot Probability  
⬇  
Final Prediction
""")

# ------------------------------------------------
# Preprocessing
# ------------------------------------------------
    st.header("🧹 Preprocessing")

    steps, clean_text = preprocess_text(tweet)

    for name,text in steps:

        with st.expander(name):
            st.code(text)

# ------------------------------------------------
# Tokenization
# ------------------------------------------------
    st.header("🔤 Tokenization")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokens = tokenizer.tokenize(clean_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    df_tokens = pd.DataFrame({
        "Token":tokens,
        "Token ID":token_ids
    })

    st.dataframe(df_tokens)

    fig = px.bar(
        df_tokens,
        x="Token",
        y="Token ID",
        title="Token ID Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Model Processing
# ------------------------------------------------
    results = []

    for model_name,(hf,weights) in MODELS.items():

        st.header(f"🤖 {model_name}")

        model, tokenizer = load_model(hf,weights)

        inputs = tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        tokens_model = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():

            prob, outputs = model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

        score = prob.item()

        prediction = "Bot 🤖" if score>0.5 else "Human 👤"

# ------------------------------------------------
# Hidden State
# ------------------------------------------------
        st.subheader("Hidden State Shape")

        st.code(str(outputs.last_hidden_state.shape))

# ------------------------------------------------
# Attention Layer Viewer
# ------------------------------------------------
        st.subheader("🔬 Transformer Layer Explorer")

        num_layers = len(outputs.attentions)

        layer = st.slider(
            f"{model_name} Layer",
            1,
            num_layers,
            1,
            key=f"{model_name}_layer"
        ) - 1

        num_heads = outputs.attentions[layer].shape[1]

        head = st.slider(
            f"{model_name} Head",
            1,
            num_heads,
            1,
            key=f"{model_name}_head"
        ) - 1

        attention = outputs.attentions[layer][0][head].detach().cpu().numpy()

        attention = attention[:len(tokens_model),:len(tokens_model)]

        fig = px.imshow(
            attention,
            x=tokens_model,
            y=tokens_model,
            title=f"{model_name} Layer {layer+1} Head {head+1}"
        )

        st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Attention Flow Animation
# ------------------------------------------------
        st.subheader("Attention Flow Across Layers")

        attention_layers = []

        for att_layer in outputs.attentions:

            attn = att_layer[0].mean(dim=0).detach().cpu().numpy()

            attn = attn[:len(tokens_model),:len(tokens_model)]

            attention_layers.append(attn)

        attention_flow = np.stack(attention_layers)

        fig = px.imshow(
            attention_flow,
            animation_frame=0,
            x=tokens_model,
            y=tokens_model,
            title=f"{model_name} Attention Flow"
        )

        st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Probability Gauge
# ------------------------------------------------
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text":"Bot Probability"},
            gauge={
                "axis":{"range":[0,1]},
                "steps":[
                    {"range":[0,0.5],"color":"green"},
                    {"range":[0.5,1],"color":"red"}
                ]
            }
        ))

        st.plotly_chart(gauge,use_container_width=True)

        results.append({
            "Model":model_name,
            "Prediction":prediction,
            "Confidence":round(score,3)
        })

# ------------------------------------------------
# Model Comparison
# ------------------------------------------------
    st.header("📊 Model Comparison")

    df = pd.DataFrame(results)

    st.dataframe(df)

    fig = px.bar(
        df,
        x="Model",
        y="Confidence",
        color="Prediction",
        text="Confidence"
    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Final Decision
# ------------------------------------------------
    st.header("🧠 Final Prediction")

    bot_votes = len(df[df["Prediction"]=="Bot 🤖"])
    human_votes = len(df[df["Prediction"]=="Human 👤"])

    if bot_votes > human_votes:
        st.error("BOT ACCOUNT DETECTED 🤖")
    else:
        st.success("HUMAN ACCOUNT DETECTED 👤")
