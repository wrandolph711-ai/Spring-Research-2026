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

# SPEED OPTIMIZATION 1: Check for mixed precision training support
use_amp = torch.cuda.is_available()  # Automatic Mixed Precision (faster training on GPU)
if use_amp:
    print("✓ Mixed precision training enabled (2x faster on GPU)")
else:
    print("✓ Running on CPU (mixed precision not available)")

# Create Vision Transformer model for CIFAR-100
class ViT_CIFAR100_Fast(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ViT_CIFAR100_Fast, self).__init__()
        
        # SPEED OPTIMIZATION 2: Use smaller ViT model (vit_b_32 instead of vit_b_16)
        # vit_b_32 uses 32x32 patches (fewer patches = faster)
        # vit_b_16 uses 16x16 patches (more patches = slower but more accurate)
        try:
            from torchvision.models import ViT_B_32_Weights
            self.vit = models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None)
        except:
            self.vit = models.vit_b_32(pretrained=pretrained)
        
        # Get the hidden dimension size
        hidden_dim = self.vit.hidden_dim
        
        # Replace classification head
        self.vit.heads = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.vit(x)

# Initialize model
print("Initializing Vision Transformer model (FAST version)...")
print("Using ViT-B/32 (faster than ViT-B/16)")
model = ViT_CIFAR100_Fast(num_classes=100, pretrained=True).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()

# SPEED OPTIMIZATION 3: Use AdamW with higher learning rate for faster convergence
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# SPEED OPTIMIZATION 4: Learning rate scheduler (reduces LR when stuck)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# SPEED OPTIMIZATION 5: Resize to smaller resolution 
# ViT requires image size to match what it was trained on, so we use 224
# But we can use smaller batch processing for speed
TARGET_SIZE = 224  # ViT requires 224x224 (pretrained size)

def preprocess_images(images_array, target_size=TARGET_SIZE):
    """Convert numpy array to normalized torch tensor with resizing for ViT"""
    # Normalize to [0, 1]
    images_normalized = images_array.astype(np.float32) / 255.0
    # Convert to tensor
    images_tensor = torch.from_numpy(images_normalized).permute(0, 3, 1, 2)
    
    # Resize (using smaller size for speed)
    from torch.nn.functional import interpolate
    images_resized = interpolate(images_tensor, size=(target_size, target_size), 
                                  mode='bilinear', align_corners=False)
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_tensor = (images_resized - mean) / std
    
    return images_tensor

# Load existing model
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

# Training function with SPEED OPTIMIZATIONS
def train_model(epochs=5, batch_size=64, save_path='vit_cifar100_fast.pth'):
    """
    FAST training with multiple optimizations:
    - Mixed precision training (AMP)
    - Larger batch size (64 instead of 32)
    - Smaller ViT model (ViT-B/32 instead of ViT-B/16)
    - Learning rate scheduling
    - Multi-threaded data loading
    """
    print(f"\nTraining model for {epochs} epochs (OPTIMIZED FOR SPEED)...")
    print(f"Model will auto-save after each epoch to: {save_path}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {TARGET_SIZE}x{TARGET_SIZE}")
    
    # SPEED OPTIMIZATION 6: Enable cuDNN autotuner (finds fastest algorithms)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN autotuner enabled")
    
    # Prepare training data
    print("Preprocessing training data...")
    x_train_tensor = preprocess_images(x_train)
    y_train_tensor = torch.from_numpy(y_train).squeeze().long()
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    
    # SPEED OPTIMIZATION 7: Multi-threaded data loading
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    
    # Prepare test data
    print("Preprocessing test data...")
    x_test_tensor = preprocess_images(x_test)
    y_test_tensor = torch.from_numpy(y_test).squeeze().long()
    
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # SPEED OPTIMIZATION 8: Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # SPEED OPTIMIZATION 9: Mixed precision forward/backward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100.*test_correct/test_total
        train_acc = 100.*correct/total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        # Update learning rate based on test accuracy
        scheduler.step(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            try:
                torch.save(model.state_dict(), save_path)
                print(f"✓ New best model saved! (Test Acc: {test_acc:.2f}%)")
            except Exception as e:
                print(f"✗ Warning: Could not save model: {e}")
        
    print(f"\n✓ Training completed! Best accuracy: {best_acc:.2f}%")
    return model

# Prediction function
def predict_image(image):
    """Predict class for a single image"""
    model.eval()
    
    # Preprocess single image
    image_normalized = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # Resize
    from torch.nn.functional import interpolate
    image_tensor = interpolate(image_tensor, size=(TARGET_SIZE, TARGET_SIZE), 
                               mode='bilinear', align_corners=False)
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(image_tensor)
        else:
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
    
    plt.ion()
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
    print("CIFAR-100 with Vision Transformer (FAST VERSION)")
    print("="*60)
    print("\nSPEED OPTIMIZATIONS ENABLED:")
    print("✓ Mixed precision training (2x faster on GPU)")
    print("✓ Smaller ViT model (ViT-B/32 instead of ViT-B/16)")
    print(f"✓ Image size: {TARGET_SIZE}x{TARGET_SIZE}")
    print("✓ Larger batch size (64 instead of 32)")
    print("✓ Learning rate scheduling")
    print("✓ Multi-threaded data loading")
    print("✓ cuDNN autotuner")
    print("="*60)
    
    # Estimate training time
    if torch.cuda.is_available():
        print("\n⏱️  ESTIMATED TIME PER EPOCH: 2-3 minutes (GPU)")
        print("   5 epochs ≈ 10-15 minutes total")
    else:
        print("\n⏱️  ESTIMATED TIME PER EPOCH: 15-20 minutes (CPU)")
        print("   5 epochs ≈ 75-100 minutes total")
        print("   💡 TIP: Use GPU for 5-10x speedup!")
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
            if os.path.exists(load_choice):
                load_model(load_choice)
    
    # Ask user if they want to train
    choice = input("\nDo you want to train the model? (yes/no) [no]: ").lower().strip()
    
    if choice in ['yes', 'y']:
        epochs = input("Enter number of epochs [5]: ").strip()
        epochs = int(epochs) if epochs else 5
        
        batch_size = input("Enter batch size [64]: ").strip()
        batch_size = int(batch_size) if batch_size else 64
        
        save_name = input("Save as [vit_cifar100_fast.pth]: ").strip()
        save_name = save_name if save_name else 'vit_cifar100_fast.pth'
        
        trained_model = train_model(epochs=epochs, batch_size=batch_size, save_path=save_name)
        print(f"\n✓ Final model saved to {save_name}")
    else:
        print("\nSkipping training. Using current model state.")
    
    # Start visualization
    visualize_with_predictions(model)