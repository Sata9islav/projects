# A neural network for text auto-completion

## Structure

```
text-autocomplete/
├── data/                  
│   └── raw_dataset.csv    
│
├── src/                    
│   ├── data_utils.py       
|   ├── next_token_dataset.py   
│   ├── lstm_model.py    
|   ├── eval_lstm.py         
|   ├── lstm_train.py     
|   └── eval_transformer_pipeline.py 
│
│
|
├── solution.ipynb   
└── requirements.txt
```

## Task

A project to investigate text auto-completion models on short user messages (tweets / tweet-like texts).

The goal is to compare classical recurrent networks (LSTM, GRU) and a transformer model in terms of the quality of text continuation generation and its suitability for use on mobile devices.

### Project goals

- Implement a training pipeline for auto-completion models:
- Compare:
 - **LSTM**
 - **GRU**
 - **Transformer (distilgpt2)** .
- Evaluate models by generation quality (ROUGE),

### Data

- Collection of short texts (tweets / messages).

### Models

1. LSTM
2. GRU
3. Transformer (distilgpt2)