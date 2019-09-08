
import numpy as np
import matplotlib.pyplot as plt

MAX_EPOCH = 50
COLOR_LIST = ['red', 'orange', 'green', 'blue', 'purple', 'black', 'gray']
BATCH_SIZE_LIST = [1, 2, 4, 8, 16, 32, 64]

epoch_list = np.arange(MAX_EPOCH) + 1

# 1. Batch Normalization
batch_loss_list = np.zeros((len(BATCH_SIZE_LIST), MAX_EPOCH))
batch_accuracy_list = np.zeros((len(BATCH_SIZE_LIST), MAX_EPOCH))

for i in range(len(BATCH_SIZE_LIST)):
    f = open('./log/batch_{}.txt'.format(BATCH_SIZE_LIST[i]), 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        epoch, loss, accuracy = line.strip().split(',')
        
        epoch = int(epoch) - 1
        loss = float(loss)
        accuracy = float(accuracy)
        
        batch_loss_list[i, epoch] = loss
        batch_accuracy_list[i, epoch] = accuracy

plt.clf()
for i in range(len(COLOR_LIST)):
    plt.plot(epoch_list, batch_loss_list[i], color = COLOR_LIST[i])
plt.legend(['BN, {} imgs/gpu'.format(i) for i in BATCH_SIZE_LIST], loc = 'upper right')
plt.title('# Batch Normalization - Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.show()
plt.savefig('./res/Batch_Normalization_Train_Loss.jpg')

plt.clf()
for i in range(len(COLOR_LIST)):
    plt.plot(epoch_list, batch_accuracy_list[i], color = COLOR_LIST[i])
plt.legend(['BN, {} imgs/gpu'.format(i) for i in BATCH_SIZE_LIST], loc = 'lower right')
plt.title('# Batch Normalization - Test Accuracy')
plt.ylim(0, 100)
plt.xlabel('epochs')
plt.ylabel('accuracy (%)')
# plt.show()
plt.savefig('./res/Batch_Normalization_Test_Accuracy.jpg')

# 2. Group Normalization
group_loss_list = np.zeros((len(BATCH_SIZE_LIST), MAX_EPOCH))
group_accuracy_list = np.zeros((len(BATCH_SIZE_LIST), MAX_EPOCH))

for i in range(len(BATCH_SIZE_LIST)):
    f = open('./log/group_{}.txt'.format(BATCH_SIZE_LIST[i]), 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        epoch, loss, accuracy = line.strip().split(',')
        
        epoch = int(epoch) - 1
        loss = float(loss)
        accuracy = float(accuracy)
        
        group_loss_list[i, epoch] = loss
        group_accuracy_list[i, epoch] = accuracy

plt.clf()
for i in range(len(COLOR_LIST)):
    plt.plot(epoch_list, group_loss_list[i], color = COLOR_LIST[i])
plt.legend(['GN, {} imgs/gpu'.format(i) for i in BATCH_SIZE_LIST], loc = 'upper right')
plt.title('# Group Normalization - Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.show()
plt.savefig('./res/Group_Normalization_Train_Loss.jpg')

plt.clf()
for i in range(len(COLOR_LIST)):
    plt.plot(epoch_list, group_accuracy_list[i], color = COLOR_LIST[i])
plt.legend(['GN, {} imgs/gpu'.format(i) for i in BATCH_SIZE_LIST], loc = 'lower right')
plt.title('# Group Normalization - Test Accuracy')
plt.ylim(0, 100)
plt.xlabel('epochs')
plt.ylabel('accuracy (%)')
# plt.show()
plt.savefig('./res/Group_Normalization_Test_Accuracy.jpg')

# 3. Comparison
# # Batch Size = 4, Train Loss
# plt.clf()

# plt.plot(epoch_list, batch_loss_list[2], color = 'red')
# plt.plot(epoch_list, group_loss_list[2], color = 'blue')

# plt.legend(['BN, 4 imgs/gpu', 'GN, 4 imgs/gpu'], loc = 'upper right')
# plt.title('# BN vs GN - Train Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# # plt.show()
# plt.savefig('./res/BN_vs_GN_Train_Loss.jpg')

# # Batch Size = 4, Test Accuracy
# plt.clf()

# plt.plot(epoch_list, batch_accuracy_list[2], color = 'red')
# plt.plot(epoch_list, group_accuracy_list[2], color = 'blue')

# plt.legend(['BN, 4 imgs/gpu', 'GN, 4 imgs/gpu'], loc = 'lower right')
# plt.title('# BN vs GN - Test Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy (%)')
# plt.ylim(0, 100)
# # plt.show()
# plt.savefig('./res/BN_vs_GN_Test_Accuracy.jpg')
