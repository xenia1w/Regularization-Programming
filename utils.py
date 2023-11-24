import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from torchvision.datasets import MNIST


def load_mnist_subset(first_n_classes: int, every_nth_sample: int = 1, transform=T.ToTensor()):
    data_root = './data'
    train_dataset = MNIST(data_root, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_root, train=False, download=True, transform=T.ToTensor())

    mean, std = train_dataset.data.float().mean(0).flatten(), train_dataset.data.float().std(0).flatten()
    selected_classes = list(range(first_n_classes))
    # Let's only select zeros and ones
    train_idx, test_idx = 0, 0
    for label in selected_classes:
        train_idx += (train_dataset.targets == label)
        test_idx += (test_dataset.targets == label)
    train_idx, test_idx = train_idx.bool(), test_idx.bool()
    

    train_dataset = TensorDataset(
        train_dataset.data[train_idx][::every_nth_sample].float().unsqueeze(1), 
        train_dataset.targets[train_idx][::every_nth_sample]
    )
    test_dataset = TensorDataset(
        test_dataset.data[test_idx].float().unsqueeze(1), 
        test_dataset.targets[test_idx]
    )
    print(f"Loaded {len(train_dataset)} training samples")
    return selected_classes, train_dataset, test_dataset


def eval_model(model, loader):
    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (100 * correct / total)

def show_single_image(img, ax=None):
    if ax is None:
        ax = plt.gca()
    if isinstance(img, torch.Tensor):
        img = img.detach()
    ax.imshow(img.permute((1,2,0)), cmap='gray')
    ax.axis('off')

def show_samples(dataset, transform=None):
    h, w = 3, 10
    fig, ax = plt.subplots(h, w)
    fig.set_size_inches((w, h))
    ax = ax.ravel()
    for i in range(h * w):
        img, label = dataset[i]    
        ax[i].imshow(img.permute((1,2,0)), cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(label.item())
    plt.tight_layout()
    plt.show()


def visualize_transform(dataset, transform):
    h, w = 3, 10
    fig, ax = plt.subplots(h, w)
    fig.set_size_inches((w, h))
    for ri in range(h):
        for ci in range(w):
            img = dataset[ri][0]
            if ci > 0:
                img = transform(img)
            ax[ri, ci].imshow(img.permute( (1,2,0)), cmap='gray')
            ax[ri, ci].axis('off')
            ax[ri, ci].set_title('Orig.' if ci == 0 else None)
            if ci == 0:
                ax[ri, ci].add_patch(plt.Rectangle((0, 0), 28, 28, linewidth=2, edgecolor='red', facecolor='none', clip_on=False))  # Add a red frame    
    plt.tight_layout()
    plt.show()

# Check Device configuration
device = torch.device('cpu')

# Define Hyper-parameters 
def train_model(model, train_dataset, test_dataset, optimizer, random_transform=lambda x:x, num_batches = 5000, verbose=True):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()


    batch_size = 128
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    # Train the model
    test_accs = [eval_model(model, test_loader)]
    train_accs = [eval_model(model, train_loader)]

    batch_counter = 0
    while batch_counter < num_batches:
        for _, (images, labels) in enumerate(train_loader):  
            batch_counter += 1
            # Move tensors to the configured device
            images = random_transform(images).reshape(-1, 28*28)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (batch_counter + 1) % (num_batches// 100) == 0:
                model.eval()
                train_accs.append(eval_model(model, train_loader))
                test_accs.append(eval_model(model, test_loader))
                model.train()
                if (batch_counter + 1) % (num_batches// 10) == 0 and verbose:
                    print ('Batch [{}/{}], Loss: {:.2e} Acc. Train: {:.3f} Test: {:.3f}' 
                       .format(batch_counter+1, num_batches, 
                               loss.item(), 
                               train_accs[-1], test_accs[-1]))
            if batch_counter >= num_batches:
                break
    
    print('Accuracy of the network on the 10000 test images: {} %'.format(test_accs[-1]))

    return train_accs, test_accs