import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm
import os
import time

# Load CIFAR-100
print("Loading CIFAR-100 dataset...")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
images = np.concatenate((x_train, x_test))
labels = np.concatenate((y_train, y_test))

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

# Superclass categories for hints
categories = {
    'animal': ['bear', 'beaver', 'cattle', 'chimpanzee', 'elephant', 'fox', 'hamster', 'kangaroo', 
               'leopard', 'lion', 'otter', 'possum', 'rabbit', 'raccoon', 'seal', 'shark', 'shrew', 
               'skunk', 'squirrel', 'tiger', 'wolf', 'camel', 'dolphin', 'mouse', 'whale'],
    'aquatic': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout', 'dolphin', 'seal', 'whale'],
    'insect': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'invertebrate': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'vehicle': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'streetcar', 'tank', 'tractor', 'train'],
    'plant': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree', 'mushroom', 
              'orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'fruit_vegetable': ['apple', 'orange', 'pear', 'sweet_pepper'],
    'household': ['bed', 'chair', 'couch', 'table', 'wardrobe', 'bottle', 'bowl', 'can', 'cup', 
                  'plate', 'clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'outdoor': ['bridge', 'castle', 'cloud', 'forest', 'house', 'mountain', 'plain', 'road', 
                'rocket', 'sea', 'skyscraper', 'lawn_mower'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptile': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
}

def get_category(class_name):
    """Get the category of a class"""
    for category, items in categories.items():
        if class_name in items:
            return category
    return 'other'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== MODEL 1: ResNet18 ====================
class ResNet18_CIFAR100(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ResNet18_CIFAR100, self).__init__()
        try:
            from torchvision.models import ResNet18_Weights
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except:
            self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# ==================== MODEL 2: VGG19 ====================
class VGG19_CIFAR100(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(VGG19_CIFAR100, self).__init__()
        try:
            from torchvision.models import VGG19_Weights
            self.vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        except:
            self.vgg19 = models.vgg19(pretrained=pretrained)
        
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        self.vgg19.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        return self.vgg19(x)

# Initialize models
print("\nInitializing models...")
model_resnet = ResNet18_CIFAR100(num_classes=100, pretrained=True).to(device)
model_vgg = VGG19_CIFAR100(num_classes=100, pretrained=True).to(device)

models_dict = {
    'ResNet18': model_resnet,
    'VGG19': model_vgg
}

print("All models loaded!")

# Preprocessing
def preprocess_images(images_array):
    images_normalized = images_array.astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images_normalized).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images_tensor - mean) / std

def train_model(model, model_name, epochs=3, batch_size=128):
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {epochs} epochs...")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    x_train_tensor = preprocess_images(x_train)
    x_test_tensor = preprocess_images(x_test)
    
    y_train_tensor = torch.from_numpy(y_train).squeeze().long()
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    y_test_tensor = torch.from_numpy(y_test).squeeze().long()
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Test
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.*correct/total
        test_acc = 100.*test_correct/test_total
        print(f"{model_name} Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    return model

def predict_image_with_top_k(image, model, k=5):
    """Get top k predictions with probabilities"""
    model.eval()
    
    image_normalized = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k)
    
    return [(class_names[idx.item()], prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]

def predict_image(image, model):
    """Get single best prediction"""
    top_predictions = predict_image_with_top_k(image, model, k=1)
    return class_names.index(top_predictions[0][0]), top_predictions[0][1]

def guessing_game():
    print("\n" + "="*60)
    print("GUESSING GAME: YOU vs ResNet18 vs VGG19!")
    print("="*60)
    
    # Choose difficulty
    print("\nCHOOSE DIFFICULTY:")
    print("1. Easy - Multiple choice (5 options) + hints available")
    print("2. Medium - Free text + hints available")
    print("3. Hard - Timed (10 seconds) + one hint only")
    print("4. Expert - Quick flash (2 seconds) + no hints")
    
    difficulty = input("\nSelect difficulty (1-4) [2]: ").strip() or "2"
    
    if difficulty == "1":
        mode = "easy"
        time_limit = None
        hints_available = 3
        print("\nEASY MODE: Multiple choice with 3 hints available!")
    elif difficulty == "3":
        mode = "hard"
        time_limit = 10
        hints_available = 1
        print("\nHARD MODE: 10 seconds to guess, 1 hint only!")
    elif difficulty == "4":
        mode = "expert"
        time_limit = None
        hints_available = 0
        flash_duration = 2
        print("\nEXPERT MODE: Image flashes for 2 seconds, no hints!")
    else:
        mode = "medium"
        time_limit = None
        hints_available = 3
        print("\nMEDIUM MODE: Free text with 3 hints available!")
    
    scores = {
        'You': 0,
        'ResNet18': 0,
        'VGG19': 0
    }
    round_num = 0
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        while True:
            round_num += 1
            idx = np.random.randint(0, len(images))
            image = images[idx]
            true_label = class_names[labels[idx][0]]
            category = get_category(true_label)
            
            hints_used = 0
            hints_left = hints_available
            
            # EXPERT MODE: Flash image briefly
            if mode == "expert":
                ax.clear()
                ax.imshow(image)
                ax.set_title(f"Round {round_num}: MEMORIZE THIS!", fontsize=14, fontweight='bold', color='red')
                ax.axis('off')
                plt.pause(0.1)
                
                print(f"\n{'='*60}")
                print(f"ROUND {round_num} - MEMORIZING...")
                print(f"{'='*60}")
                
                for i in range(flash_duration, 0, -1):
                    print(f"Image visible for {i} more seconds...")
                    time.sleep(1)
                
                # Hide image
                ax.clear()
                ax.text(0.5, 0.5, "IMAGE HIDDEN\n\nWhat was it?", 
                       ha='center', va='center', fontsize=20, fontweight='bold')
                ax.axis('off')
                plt.pause(0.1)
            else:
                # Show image normally
                ax.clear()
                ax.imshow(image)
                ax.set_title(f"Round {round_num}: What is this?", fontsize=14, fontweight='bold')
                ax.axis('off')
                plt.pause(0.1)
            
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}")
            print(f"{'='*60}")
            
            # Get AI predictions (for hints and scoring)
            top_5_resnet = predict_image_with_top_k(image, models_dict['ResNet18'], k=5)
            top_5_vgg = predict_image_with_top_k(image, models_dict['VGG19'], k=5)
            
            # EASY MODE: Show multiple choice
            if mode == "easy":
                # Create options: correct answer + 4 random wrong answers
                options = [true_label]
                while len(options) < 5:
                    random_class = class_names[np.random.randint(0, len(class_names))]
                    if random_class not in options:
                        options.append(random_class)
                np.random.shuffle(options)
                
                print("\nOPTIONS:")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                
                if hints_left > 0:
                    print(f"\nHINTS AVAILABLE: {hints_left}")
                    print("Type 'hint' to use a hint, or enter your answer (1-5)")
                
                while True:
                    if time_limit:
                        start_time = time.time()
                        your_guess = input(f"\nYour answer (1-5, or 'hint') [{time_limit}s]: ").strip().lower()
                        elapsed = time.time() - start_time
                        if elapsed > time_limit:
                            print(f"TIME'S UP! ({elapsed:.1f}s)")
                            your_guess = ""
                            break
                    else:
                        your_guess = input("\nYour answer (1-5, or 'hint'): ").strip().lower()
                    
                    if your_guess == 'hint' and hints_left > 0:
                        hints_left -= 1
                        hints_used += 1
                        
                        if hints_used == 1:
                            print(f"\nHINT 1: Category is '{category}'")
                        elif hints_used == 2:
                            # 50/50 hint - remove 2 wrong answers
                            wrong_options = [opt for opt in options if opt != true_label]
                            np.random.shuffle(wrong_options)
                            remove_these = wrong_options[:2]
                            print(f"\nHINT 2 (50/50): It's NOT '{remove_these[0]}' or '{remove_these[1]}'")
                        elif hints_used == 3:
                            print(f"\nHINT 3: ResNet18 thinks it's '{top_5_resnet[0][0]}' ({top_5_resnet[0][1]*100:.1f}%)")
                        
                        print(f"Hints remaining: {hints_left}")
                    else:
                        break
                
                # Convert number to class name
                if your_guess.isdigit() and 1 <= int(your_guess) <= 5:
                    your_guess = options[int(your_guess) - 1]
                else:
                    your_guess = ""
            
            # MEDIUM/HARD MODE: Free text
            else:
                if hints_left > 0:
                    print(f"\nHINTS AVAILABLE: {hints_left}")
                    print("Type 'hint' for a hint, or enter your guess")
                
                while True:
                    if time_limit:
                        start_time = time.time()
                        your_guess = input(f"\nYour guess (or 'hint') [{time_limit}s]: ").strip().lower()
                        elapsed = time.time() - start_time
                        if elapsed > time_limit:
                            print(f"TIME'S UP! ({elapsed:.1f}s)")
                            your_guess = ""
                            break
                    else:
                        your_guess = input("\nYour guess (or 'hint'): ").strip().lower()
                    
                    if your_guess == 'hint' and hints_left > 0:
                        hints_left -= 1
                        hints_used += 1
                        
                        if hints_used == 1:
                            print(f"\nHINT 1: Category is '{category}'")
                        elif hints_used == 2:
                            print(f"\nHINT 2: Top 3 guesses from ResNet18:")
                            for i, (label, conf) in enumerate(top_5_resnet[:3], 1):
                                print(f"  {i}. {label} ({conf*100:.1f}%)")
                        elif hints_used == 3:
                            first_letter = true_label[0]
                            print(f"\nHINT 3: First letter is '{first_letter.upper()}'")
                        
                        print(f"Hints remaining: {hints_left}")
                    else:
                        break
            
            # Get AI predictions
            resnet_guess = top_5_resnet[0][0]
            vgg_guess = top_5_vgg[0][0]
            
            print(f"\nAI PREDICTIONS:")
            print(f"   ResNet18: {resnet_guess} ({top_5_resnet[0][1]*100:.1f}%)")
            print(f"   VGG19: {vgg_guess} ({top_5_vgg[0][1]*100:.1f}%)")
            
            input("\nPress Enter to see the answer...")
            
            # Show answer
            print(f"\nCORRECT ANSWER: {true_label}")
            print(f"Category: {category}")
            
            # Score everyone
            you_correct = your_guess == true_label.lower().replace('_', ' ')
            
            # Point deductions for hints
            points_earned = 1
            if hints_used == 1:
                points_earned = 0.75
            elif hints_used == 2:
                points_earned = 0.5
            elif hints_used >= 3:
                points_earned = 0.25
            
            if you_correct:
                scores['You'] += points_earned
                if hints_used > 0:
                    print(f"YOU got it right! (+{points_earned:.2f} points, used {hints_used} hints)")
                else:
                    print(f"YOU got it right! (+{points_earned} point, no hints!)")
            else:
                print("You got it wrong")
            
            if resnet_guess == true_label:
                scores['ResNet18'] += 1
                print("ResNet18 got it right!")
            else:
                print("ResNet18 got it wrong")
            
            if vgg_guess == true_label:
                scores['VGG19'] += 1
                print("VGG19 got it right!")
            else:
                print("VGG19 got it wrong")
            
            # Update display
            ax.clear()
            ax.imshow(image)
            
            result_text = f"Answer: {true_label} ({category})\n\n"
            result_text += f"Your guess: {your_guess if your_guess else 'NO ANSWER'} {'CORRECT' if you_correct else 'WRONG'}"
            if hints_used > 0:
                result_text += f" ({hints_used} hints)\n"
            else:
                result_text += "\n"
            result_text += f"ResNet18: {resnet_guess} {'CORRECT' if resnet_guess==true_label else 'WRONG'}\n"
            result_text += f"VGG19: {vgg_guess} {'CORRECT' if vgg_guess==true_label else 'WRONG'}"
            
            ax.set_title(result_text, fontsize=10)
            ax.axis('off')
            
            print(f"\nSCOREBOARD:")
            print(f"   You: {scores['You']:.2f}")
            print(f"   ResNet18: {scores['ResNet18']}")
            print(f"   VGG19: {scores['VGG19']}")
            print(f"{'='*60}")
            
            plt.pause(2)
            
            next_round = input("\nNext round? (yes/no) [yes]: ").strip().lower()
            if next_round in ['no', 'n', 'quit', 'exit']:
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        plt.close()
        print(f"\n{'='*60}")
        print("FINAL RESULTS!")
        print(f"{'='*60}")
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(sorted_scores, 1):
            if name == 'You':
                print(f"{i}. {name}: {score:.2f}/{round_num} ({100*score/round_num:.1f}%)")
            else:
                print(f"{i}. {name}: {score}/{round_num} ({100*score/round_num:.1f}%)")
        
        winner = sorted_scores[0][0]
        if winner == 'You':
            print("\nYOU WIN! You beat both AI models!")
        else:
            print(f"\n{winner} wins! Better luck next time!")
        print(f"{'='*60}\n")

# Main
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CIFAR-100: YOU vs ResNet18 vs VGG19")
    print("="*60)
    
    # Load saved models
    for model_name in ['ResNet18', 'VGG19']:
        filename = f"{model_name.lower()}_cifar100.pth"
        if os.path.exists(filename):
            try:
                models_dict[model_name].load_state_dict(torch.load(filename, map_location=device))
                print(f"Loaded {model_name} from {filename}")
            except:
                print(f"Failed to load {model_name}")
    
    # Train models?
    choice = input("\nTrain models? (yes/no) [no]: ").lower().strip()
    if choice in ['yes', 'y']:
        train_choice = input("Train which models? (all/resnet/vgg) [all]: ").lower().strip()
        epochs = int(input("Epochs [3]: ") or "3")
        
        if train_choice in ['all', '']:
            for model_name, model in models_dict.items():
                train_model(model, model_name, epochs=epochs)
                torch.save(model.state_dict(), f"{model_name.lower()}_cifar100.pth")
                print(f"{model_name} saved!")
        elif train_choice == 'resnet':
            train_model(models_dict['ResNet18'], 'ResNet18', epochs=epochs)
            torch.save(models_dict['ResNet18'].state_dict(), "resnet18_cifar100.pth")
        elif train_choice == 'vgg':
            train_model(models_dict['VGG19'], 'VGG19', epochs=epochs)
            torch.save(models_dict['VGG19'].state_dict(), "vgg19_cifar100.pth")
    
    # Play game
    print("\n" + "="*60)
    print("Ready to play!")
    print("="*60)
    input("Press Enter to start the game...")
    guessing_game()