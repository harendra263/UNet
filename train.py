from dataloader import trainloader
from loss import criterion, optimizer
from cnn import net
import torch

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i+ 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)