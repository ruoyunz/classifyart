from util import *

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomCrop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_FOLDS = 5
FOLD = 0
data_dir = "../imet-2020-fgvc7/"

# model params
batch_size = 128
n_epochs = 10
lr = 1e-4
thresh = 0.05

SEED = 1

# Seeding step
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['id'].values[idx]
        file_path = f'{data_dir}train/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)

        label = self.df.iloc[idx]
        target = torch.zeros(N_CLS)
        for cls in label.attribute_ids.split():
            target[int(cls)] = 1
        
        return image, target
    
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['id'].values[idx]
        file_path = f'{data_dir}test/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)

        return image

def train(model, device, train_loader, optimizer, loss_fn, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    epoch_loss = 0

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        
        loss = loss_fn(output, target)      # Compute loss
        loss = loss.sum() / loss.shape[0]
        
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step

        epoch_loss += loss.item() 
    
    return epoch_loss / len(train_loader)


def test(model, device, test_loader, loss_fn):
    model.eval()    # Set the model to inference mode
    test_loss = 0

    preds = []
    new_labels = []
    
    with torch.no_grad():   # For the inference step, gradient is not computed
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            preds.append(torch.sigmoid(output).to('cpu').numpy())
            new_labels.append(target.to('cpu').numpy())

            loss = loss_fn(output, target)      # Compute loss
            loss = loss.sum() / loss.shape[0]
        
            test_loss += loss.item() 
    
    preds = np.concatenate(preds)
    new_labels = np.concatenate(new_labels)
    score = fbeta_score(np.asarray(new_labels), predictions(np.asarray(preds), thresh), beta=2, average='samples')

    return test_loss / len(test_loader), score

def get_model():
    model = models.resnet18(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, N_CLS)
    return model

def main():
    data_dir = "../imet-2020-fgvc7/"

    train_data = pd.read_csv(data_dir + 'train.csv')
    labels = pd.read_csv(data_dir + 'labels.csv')

    cls_counts = Counter(cls for classes in train_data['attribute_ids'].str.split() for cls in classes)

    seed_all(1)

    # use train_data.copy() if not using subset of data
    folds = train_data.sample(n=1000, random_state=SEED).reset_index(drop=True).copy()
    folds = make_folds(folds, N_FOLDS, SEED)
    
    # split into training, validation, and test 
    train_idx = folds[folds['fold'] == 2].index
    for i in range(3, N_FOLDS):
        train_idx = train_idx.union(folds[folds['fold'] == i].index)
    
    val_idx = folds[folds['fold'] == 1].index

    test_idx = folds[folds['fold'] == 0].index
    np.save("test_idx.npy", test_idx)


    train_dataset = TrainDataset(folds.loc[train_idx].reset_index(drop=True), 
                                 folds.loc[train_idx]['attribute_ids'], 
                                 transform=Compose([ Resize((H, W)), ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),]))
    
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), 
                                 folds.loc[val_idx]['attribute_ids'], 
                                 transform=Compose([ Resize((H, W)), ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),]))
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = get_model()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    best_score = 0.
    best_thresh = 0.
    best_loss = np.inf

    for epoch in range(n_epochs):
        t0 = time.time()
        # Train
        avg_loss = train(model, device, train_loader, optimizer, loss_fn, epoch)
        # Validate
        avg_val_loss, score = test(model, device, valid_loader, loss_fn)

        dt = time.time() - t0
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {dt:.0f}s  threshold: {thresh}  f2_score: {score}')

        if score > best_score:
            best_score = score
            best_thresh = thresh
            print('Saving best f2 score model.')
            torch.save(model.state_dict(), 'best_f2.pth')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print('Saving best validation loss model.')
            torch.save(model.state_dict(), 'best_val.pth')


if __name__ == '__main__':
    main()





