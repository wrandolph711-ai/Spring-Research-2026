import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from tqdm import tqdm
import os

# Load CIFAR-100 (fine labels = 100 classes)
print("Loading CIFAR-100 dataset...")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

# Combine all images
images = np.concatenate((x_train, x_test))
labels = np.concatenate((y_train, y_test))

# 100 class names
class_names = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create VGG19 model with modifications for CIFAR-100
class VGG19_CIFAR100(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(VGG19_CIFAR100, self).__init__()
        
        # Load pre-trained VGG19
        self.vgg19 = models.vgg19(pretrained=pretrained)
        
        # Modify the classifier for CIFAR-100 (100 classes)
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        # Add adaptive pooling before classifier to handle 32x32 images
        self.vgg19.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        return self.vgg19(x)

# Initialize model
print("Initializing VGG19 model...")
model = VGG19_CIFAR100(num_classes=100, pretrained=True).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def preprocess_images(images_array):
    """Convert numpy array to normalized torch tensor"""
    images_normalized = images_array.astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images_normalized).permute(0, 3, 1, 2)
    
    # Apply normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_tensor = (images_tensor - mean) / std
    
    return images_tensor

# Load existing model if available
def load_model(model_path):
    """Load a previously saved model"""
    if os.path.exists(model_path):
        print(f"\nFound saved model: {model_path}")
        load_choice = input("Do you want to load this model? (yes/no) [yes]: ").lower().strip()
        
        if load_choice not in ['no', 'n']:
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("✓ Model loaded successfully!")
                return True
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                print("Starting with fresh model instead.")
                return False
    return False

# Training function with auto-save after each epoch
def train_model(epochs=10, batch_size=64, save_path='vgg19_cifar100.pth'):
    print(f"\nTraining model for {epochs} epochs...")
    print(f"Model will auto-save after each epoch to: {save_path}")
    
    # Prepare training data
    x_train_tensor = preprocess_images(x_train)
    y_train_tensor = torch.from_numpy(y_train).squeeze().long()
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Prepare test data
    x_test_tensor = preprocess_images(x_test)
    y_test_tensor = torch.from_numpy(y_test).squeeze().long()
    
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100.*test_correct/test_total
        train_acc = 100.*correct/total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        # AUTO-SAVE after each epoch
        try:
            torch.save(model.state_dict(), save_path)
            print(f"✓ Model auto-saved to {save_path}")
        except Exception as e:
            print(f"✗ Warning: Could not save model: {e}")
    
    print("\n✓ Training completed!")
    return model

# Prediction function
def predict_image(image):
    """Predict class for a single image"""
    model.eval()
    
    # Preprocess single image
    image_normalized = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # Apply VGG normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

# Visualization with predictions
def visualize_with_predictions(trained_model=None):
    """Visualize random images with true labels and predictions"""
    global model
    if trained_model is not None:
        model = trained_model
    
    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    
    print("\nStarting visualization... Press Ctrl+C to stop")
    
    try:
        while True:
            idx = np.random.randint(0, len(images))
            
            image = images[idx]
            true_label = class_names[labels[idx][0]]
            
            # Get prediction
            pred_idx, confidence = predict_image(image)
            pred_label = class_names[pred_idx]
            
            ax.clear()
            ax.imshow(image)
            
            # Color code: green if correct, red if wrong
            color = 'green' if pred_label == true_label else 'red'
            
            title = f"True: {true_label}\nPredicted: {pred_label} ({confidence*100:.1f}%)"
            ax.set_title(title, color=color, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            plt.pause(2)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
        plt.close()

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CIFAR-100 Classification with VGG19")
    print("="*60)
    
    # Check for existing models
    print("\nLooking for saved models...")
    saved_models = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if saved_models:
        print("\nFound saved models:")
        for i, model_file in enumerate(saved_models, 1):
            print(f"  {i}. {model_file}")
        
        load_choice = input("\nLoad a saved model? Enter number or 'no' [no]: ").strip()
        
        if load_choice.isdigit() and 1 <= int(load_choice) <= len(saved_models):
            model_to_load = saved_models[int(load_choice) - 1]
            load_model(model_to_load)
        elif load_choice.lower() not in ['no', 'n', '']:
            # Try to load by filename
            if os.path.exists(load_choice):
                load_model(load_choice)
    
    # Ask user if they want to train
    choice = input("\nDo you want to train the model? (yes/no) [no]: ").lower().strip()
    
    if choice in ['yes', 'y']:
        epochs = input("Enter number of epochs [10]: ").strip()
        epochs = int(epochs) if epochs else 10
        
        save_name = input("Save as [vgg19_cifar100.pth]: ").strip()
        save_name = save_name if save_name else 'vgg19_cifar100.pth'
        
        trained_model = train_model(epochs=epochs, batch_size=64, save_path=save_name)
        print(f"\n✓ Final model saved to {save_name}")
    else:
        print("\nSkipping training. Using current model state.")
    
    # Start visualization
    visualize_with_predictions(model)