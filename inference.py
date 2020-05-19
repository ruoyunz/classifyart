from baseline import *


def main():
    data_dir = "../imet-2020-fgvc7/"

    train_data = pd.read_csv(data_dir + 'train.csv')
    labels = pd.read_csv(data_dir + 'labels.csv')

    cls_counts = Counter(cls for classes in train_data['attribute_ids'].str.split() for cls in classes)

    seed_all(1)

    # use train_data.copy() if not using subset of data
    folds = train_data.sample(n=1000, random_state=SEED).reset_index(drop=True).copy()
    folds = make_folds(folds, N_FOLDS, SEED)
    
    test_idx = folds[folds['fold'] == 0].index

    test_dataset = TrainDataset(folds.loc[test_idx].reset_index(drop=True), 
                                 folds.loc[test_idx]['attribute_ids'], 
                                 transform=Compose([ Resize((H, W)), ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),]))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model()

    weights_path = 'best_score.pth'
    model.load_state_dict(torch.load(weights_path))
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    model.to(device) 

    preds = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)): 
            data = data.to(device)
            output = model(data)
            
            preds.append(torch.sigmoid(output).to('cpu').numpy())

    threshold = 0.05
    predictions = np.concatenate(preds) > threshold

    print(test(model, device, test_loader, loss_fn))


if __name__ == '__main__':
    main()
