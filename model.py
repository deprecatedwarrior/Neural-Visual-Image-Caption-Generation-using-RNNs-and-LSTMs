import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
    
        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # Define the LSTM
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            #dropout=0.2,
                            batch_first=True)
        # Define the final, fully-connected output layer that maps the hidden state output to vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        

        
        
    def forward(self, features, captions):
        # create embedding vectors for each caption 
        embeds = self.word_embeddings(captions[:,:-1])
        
        # Concatenate the features and caption inputs
        embeddings = torch.cat((features.unsqueeze(1), embeds), 1)
        
        lstm_out, self.hidden = self.lstm(embeddings)
        print (lstm_out.shape)
        
        # Convert LSTM outputs to word predictions
        outputs = self.fc(lstm_out)
        
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # initialize the hidden states
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        preds = []
        
        # Feed output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc(lstm_out) 
            outputs = outputs.squeeze(1) 
            wordid  = outputs.argmax(dim=1)     # find the word with the max probability
            preds.append(wordid.item())
    
            # prepare the inputs for the next word
            inputs = self.word_embeddings(wordid).unsqueeze(1)

        return preds
    