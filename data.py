import matplotlib.pyplot as plt
import numpy as np
import dataloader
import torchvision


# functions to show an image

def imshow(img):
    img = img /2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random images
dataiter = iter(dataloader.trainloader)
images, labels = dataiter.next()

#Show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % dataloader.classes[labels[j]] for j in range(dataloader.BATCH_SIZE)))