import torch
import torchvision
import torchtext
print(torch.__version__)
print(torchvision.__version__)
print(torchtext.__version__)
from models.ops import MultiScaleDeformableAttention
import MedVit_Adapter
from MedVit_adapter import MedViT_adapter_small as small

import medmnist
from medmnist import INFO 
data_flag = 'retinamnist'
# [tissuemnist , pathmnist, chestmnist, dermamnist, octmnisr ,pnemonismnist , retinamnist, bloodmnist, tissuemnist, organcmist, organs ]
download = True

NUM_EPOCHS = 15
BATCH_SIZE = 15
LR = 0.005

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

import torchvision.transforms as transforms
#preprocessing
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda image:image.convert('RGB')),
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]
)
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
    
import torch.utils.data as data

# load the data
train_dataset = DataClass(split='train', transform=train_transform, download=download)
val_dataset = DataClass(split='val', transform=train_transform,download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)


# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("+++++++++++++++++")
print(test_dataset)

model = small()

model.proj_head[0] = nn.Linear(in_features=1024, out_featuers=n_classes, bais=True)

#define loss function and optimizer
if task == 'multi-task, binaty-class':
    criterion = nn.BCEWihtLogLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.functional import softmax
from tqdm import tqdm
# training
for epoch in range(NUM_EPOCHS):

  model = model.to(device)
  model.train()

  for inputs, targets in tqdm(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs.to(torch.float32))

    if task == 'multi-label, binary-class':
      targets = targets.to(torch.float32).unsqueeze(1)
    else:
      targets = targets.to(torch.long)
      targets = targets.view(-1)


    predicted_classes = torch.argmax(outputs, dim=1)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Delete tensors to free up memory
    del inputs, targets, outputs, predicted_classes
    # Empty the cache to clear up some more memory
    torch.cuda.empty_cache()


from sklearn.metrics import roc_auc_score, accuracy_score

# switch to evaluation mode
def test(split):
  model.eval()
  # Lists to store actual and predicted values
  actuals = []
  probas = []
  predictions = []

  if split == 'val':
        data_loader = train_loader_at_eval
  else:
        data_loader = test_loader
        
  with torch.no_grad():
      for inputs, targets in tqdm(data_loader):

          inputs, targets = inputs.to(device), targets.to(device)

          if task == 'multi-label, binary-class':
              targets = targets.to(torch.float32).unsqueeze(1)
          else:
              targets = targets.to(torch.long)
              targets = targets.view(-1)

          outputs = model(inputs.to(torch.float32))
          softmax_outputs = softmax(outputs, dim=1)

          _, predicted_classes = torch.max(outputs, 1)

          # Store the actual targets and predicted probabilities
          actuals.extend(targets.cpu().numpy())
          probas.extend(softmax_outputs.detach().cpu().numpy())
          # Probability of positive class
          predictions.extend(predicted_classes.cpu().numpy())
  print('\n')
  auc = roc_auc_score(actuals, probas, multi_class='ovr')
  accuracy = accuracy_score(actuals, predictions)

  print('AUC of the model:', auc)
  print('Accuracy of the model:', accuracy)

print('==> Evaluating...')
test('val')
test('test')