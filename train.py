#Train.py contains the code to train and save the network
print('Executing train.py....')
#import files
import argparse
import seaborn as sb
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


########### Step 1: Get Model Hyperparameters
parser = argparse.ArgumentParser(description = 'Pass in the Train and Predict Parameters')
parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'The location of the data files')
parser.add_argument('--save_dir', type = str, default = './', help = 'The location of the saved files')
parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'densenet121', 'alexnet'], help = 'Type in the prefererd Model Architecture')
parser.add_argument('--epochs', type = int, default = 10, help = 'Type in the number of epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'Type in the learning rate')
parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'], help = 'Type GPU or CPU with uppercase')
parser.add_argument('--input_layers', type = int, default = 25088, help = 'input layers, call multiple times to add input units')
parser.add_argument('--hidden_layers', type = int, default = 4096, help = 'hidden layers, call multiple times to add hidden units')
parser.add_argument('--output_layers', type = int, default = 102, help = 'output layers, call multiple times to add output units')
parser.add_argument('--drop_rate', type = float, default = 0.5, help = 'Type in the number of drop_rate')
parser.add_argument('--topk', type = int, default = 3, help = 'Type in the number of topk comparisons')
args = parser.parse_args()

hyperparameters = { 'data_dir': args.data_dir,
                    'save_dir': args.save_dir,
                    'arch': args.arch,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'gpu': args.gpu,
                    'input_layers': args.input_layers,
                    'hidden_layers': args.hidden_layers,
                    'output_layers': args.output_layers,
                    'drop': args.drop_rate,
                    'topk':args.topk
}


########## Step 2: Get Data
train_dir = hyperparameters['data_dir'] + '/train'
valid_dir = hyperparameters['data_dir'] + '/valid'
test_dir = hyperparameters['data_dir'] + '/test'

data_transforms = {
'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
'test': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),
'valid': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])}
#Directory Dictionary
directories = {'train': train_dir,
               'test': test_dir,
               'valid': valid_dir}

images_datasets = {x: datasets.ImageFolder(directories[x],transform = data_transforms[x]) 
                  for x in list(data_transforms.keys())}

dataloaders = {
'trainloader' : torch.utils.data.DataLoader(images_datasets['train'], batch_size=64, shuffle=True),
'testloader' : torch.utils.data.DataLoader(images_datasets['test'], batch_size=64, shuffle=False),
'validloader' : torch.utils.data.DataLoader(images_datasets['valid'], batch_size=64, shuffle=True)
}
#Sample checking data
#print(list(data_transforms.keys()))
#size_of_datasets = {x: len(images_datasets[x]) for x in list(data_transforms.keys()) }
#print(size_of_datasets)
#names_of_classes = images_datasets['train'].classes
#print('number of names_of_classes: '+ str(len(names_of_classes)))

#test image load
images, labels = next(iter(dataloaders['testloader']))
print('number of testloader data: '+ str(len(images[0,2])))

########## 3. Build the Network
def get_model(model_arch):
    #load a pretrained model
    if (model_arch == 'vgg16'):
        model = models.vgg16(pretrained = True)
    elif (model_arch == 'densenet121'):
        model = models.densenet121(pretrained = True)
    elif (model_arch == 'alexnet'):
        model = models.alexnet(pretrained = True)
    #print('\n going to build through the selected model next!')
    #print('\npre-loaded selected model architecture:')
    #print(model)
    return model
#end get_model function

def build_model(model, model_arch, drop_out):
    #print('\nrunning build_model function')
    for param in model.parameters():
        param.requires_grad = False

    if (model_arch == 'vgg16'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 4096)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('fc2', nn.Linear(4096, 102)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'densenet121'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 102)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'alexnet'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(9216, 4096)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('fc2', nn.Linear(4096, 102)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    else:
        print('you screwed up badly if the codes come here')
    #print(classifier)
    return classifier
#end build_model function

model = get_model(hyperparameters['arch'].lower())
model_classifier = build_model(model, hyperparameters['arch'].lower(), hyperparameters['drop'])
model.classifier = model_classifier
print('\nprinting the selected architecture ' + hyperparameters['arch'] + ' classifier = ')
print(model.classifier)

########## 4. Train the Network
def train_model(model, criterion, optimizer, epochs, load_train_data, load_valid_data, gpu):
    model.train()
    print_every = 30
    steps = 0
    use_gpu = False
    
    # change to cuda
    if (gpu == 'GPU' and torch.cuda.is_available()):
        use_gpu = True
        print('training moves to cuda')
        model.to('cuda')
    elif (gpu == 'CPU'):
        print('training moves to CPU')
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
    
        for ii, (inputs, labels) in enumerate(load_train_data):
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                accuracy = 0
                test_loss = 0

                for images, labels in iter(load_valid_data):
                    images, labels = images.to('cuda'), labels.to('cuda')
                    output = model.forward(images)
                    test_loss += criterion(output, labels).item()
                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()

                with torch.no_grad():
                #test_loss, accuracy = validation(model, testloader, criterion)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(load_valid_data)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(load_valid_data)))
                    running_loss = 0
                model.train()
#end train_model function

def check_accuracy_on_test(model, testloader, gpu):    
    correct = 0
    total = 0
    
    model.eval()
    
    if (gpu == 'GPU'):
        model.to('cuda:0')
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if (gpu == 'GPU'):
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))
#end function

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), hyperparameters['lr'])
train_model(model, criterion, optimizer, hyperparameters['epochs'], dataloaders['trainloader'], dataloaders['validloader'], hyperparameters['gpu'])
check_accuracy_on_test(model, dataloaders['testloader'], hyperparameters['gpu'])

########### 5. Save the model and weights
model.class_to_idx = images_datasets['train'].class_to_idx
checkpoint = {
    'arch': hyperparameters['arch'],
    'class_to_idx': model.class_to_idx, 
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'input_layers': hyperparameters['input_layers'],
    'hidden_layers': hyperparameters['hidden_layers'],
    'output_layers': hyperparameters['output_layers'],
    'learning rate': hyperparameters['lr'],
    'dropout': hyperparameters['drop'],
    'epochs': hyperparameters['epochs'],
    'topk': hyperparameters['topk']
}
torch.save(checkpoint, 'checkpoint.pth')
#print(checkpoint)