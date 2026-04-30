import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100

# Load CIFAR-100 (fine labels = 100 classes)
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
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple tree','motorcycle','mountain',
    'mouse','mushroom','oak tree','orange','orchid','otter','palm_tree','pear','pickup truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

plt.ion()  # interactive mode
fig, ax = plt.subplots()

while True:
    idx = np.random.randint(0, len(images))

    image = images[idx]
    label = class_names[labels[idx][0]]

    ax.clear()
    ax.imshow(image)
    ax.set_title(f"Class: {label}")
    ax.axis('off')

    plt.pause(2)