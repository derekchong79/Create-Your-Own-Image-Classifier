#This is to create an executable predict.py to run the NN for image classification.
print('Executing predict.py...')
# Imports here
import argparse
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image

# Get Model Hyperparameters
parser = argparse.ArgumentParser(description = 'Pass in the Train and Predict Parameters')
parser.add_argument('--image_path_and_name', type = str, default = './flowers/test/1/image_06764.jpg', help = 'The location of the image file and name')
parser.add_argument('--checkpoint_path_and_name', type = str, default = './checkpoint.pth', help = 'The location of the checkpoint file and name')
parser.add_argument('--category_file_path_and_name', type = str, default = './cat_to_name.json', help = 'The location of the category mapping file and name')
parser.add_argument('--topk', type = int, default = 3, help = 'Type in the number of topk comparisons')
parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'], help = 'Type GPU or CPU with uppercase')
args = parser.parse_args()

hyperparameters = { 'image_dir': args.image_path_and_name,
                    'checkpoint_dir': args.checkpoint_path_and_name,
                    'category_dir': args.category_file_path_and_name,
                    'topk': args.topk,
                    'gpu': args.gpu
}

#Label mapping
cat_file_name = hyperparameters['category_dir']
with open(cat_file_name, 'r') as f:
    cat_to_name = json.load(f)
print('number of cat_to_name:' + str(len(cat_to_name)))

########### 6. load the model and weights

def load_model(checkpoint):

    output_layers = checkpoint['output_layers']
    print('\noutput_layers = ' + str(output_layers))
    #hidden_layers = [int(x) for x in checkpoint['hidden_layers']] 
    hidden_layers = checkpoint['hidden_layers']
    input_layers = checkpoint['input_layers']
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif (model_arch == 'densenet121'):
        model = models.densenet121(pretrained = True)
    elif (model_arch == 'alexnet'):
        model = models.alexnet(pretrained = True)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    
    # Create the classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_layers, hidden_layers)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers, output_layers)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
#end of load_model function

checkpoint = torch.load(hyperparameters['checkpoint_dir'])
model = load_model(checkpoint)
#print(model)

########### 7. Process Image
def process_images(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #editing image
    image_transformer = transforms.Compose([  
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])
                                       #transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                       #                     std = [0.229, 0.224, 0.225])])
    #open the image
    prep_image = Image.open(image_path)
    
    #process image and set to float
    prep_image = image_transformer(prep_image)
    
    #make image a numpy array
    np_image = np.array(prep_image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    #transpose the image
    #prep_image = prep_image.transpose((2,0,1))
    
    return np_image
#end process_images

image_path = hyperparameters['image_dir']
image = process_images(image_path)

########### 8. Do a class prediction
def predict(image_path, model, topk_number, cat_to_name, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_images(image_path)
    tensor_image = torch.from_numpy(image).type(torch.FloatTensor)
    
    gpu_mode = False
    if (gpu == 'GPU' and torch.cuda.is_available()):
        gpu_mode = True 
        print('training moves to cuda')
        model.to('cuda')
        tensor_image = tensor_image.cuda()    
       
    elif (gpu == 'CPU'):
        print('training moves to CPU')
        model.to('cpu')
        
    # add 1 to tensor image
    tensor_image = tensor_image.unsqueeze_(0)
        
    #Probability
    probability = torch.exp(model.forward(tensor_image))
    
    #pick top probabilities
    top_probs, top_classes = probability.topk(topk_number)
    
    if gpu_mode:
        top_probs = top_probs.cpu().detach().numpy().tolist()[0]
        top_classes = top_classes.cpu().detach().numpy().tolist()[0]
    else:
        top_probs = top_probs.detach().numpy().tolist()[0]
        top_classes = top_classes.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_labels = [idx_to_class[classes] for classes in top_classes]
    top_flowers = [cat_to_name[idx_to_class[classes]] for classes in top_classes] 
    
    return top_probs, top_flowers
#end function predict

top_probs, top_flowers = predict(image_path, model, hyperparameters['topk'], cat_to_name, hyperparameters['gpu'] )
print("\nTop flowers prediction:")
print(top_flowers)
print('\ntheir probabilities:')
print(top_probs)