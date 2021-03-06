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
        
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size= embed_size,
                            hidden_size= hidden_size, 
                            num_layers= num_layers,
                            batch_first= True)
        self.dense_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, features, captions):
        captions = self.embed_layer(captions[:,:-1])
        embed = torch.cat((features.unsqueeze(1), captions), dim=1)

        out, _ = self.lstm(embed)
        outputs = self.dense_layer(out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Accepts pre-processed image tensor (inputs) and returns predicted sentence 
        (list of tensor ids of length max_len) 
        """
        res = list() 
        for i in range(max_len):
            out, states = self.lstm(inputs, states)         
            outputs = self.dense_layer(out.squeeze(1))       
            _, predicted = outputs.max(dim=1)                    
            res.append(predicted.item())
            
            inputs = self.embed_layer(predicted)             
            inputs = inputs.unsqueeze(1)                         
        return res