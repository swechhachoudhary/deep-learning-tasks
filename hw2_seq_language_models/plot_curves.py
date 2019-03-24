import numpy as np
from parse import compile
import matplotlib.pyplot as plt

for exp in range(1, 16):

    data_dict = np.load('npy_files/learning_curves_' + str(exp) + '.npy')[()]

    plt.figure()
    plt.plot(data_dict['train_ppls'], label='Train ppls')
    plt.plot(data_dict['val_ppls'], label='Val ppls')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()
    plt.title("Exp " + str(exp) + ": Learning curves of PPL over epochs")
    plt.savefig('plots/' + str(exp) + '_PPL_epoch.png')
    plt.close()

    p = compile("epoch:{}trainppl:{}valppl:{}bestval:{}time(s)spentinepoch:{:f}")
    clock_time = []
    epochs = []
    with open('logs/log_' + str(exp) + '.txt', 'r') as myfile:

        for line in myfile:
            line = line.replace('\t', '').replace(' ', '')
            result = p.parse(line)
            epochs.append(result[0])
            clock_time.append(result[4])

        clock_time = np.array(clock_time, dtype=np.float32)
        for i in range(1, clock_time.shape[0]):
            clock_time[i] += clock_time[i - 1]
        # print(clock_time)
        # print(epochs)

    plt.figure()
    plt.plot(clock_time, data_dict['train_ppls'], label='Train ppls')
    plt.plot(clock_time, data_dict['val_ppls'], label='Val ppls')
    plt.xlabel('Wall-clock-time')
    plt.ylabel('PPL')
    plt.legend()
    plt.title("Exp " + str(exp) + ": Learning curves of PPL over wall-clock-time")
    plt.savefig('plots/' + str(exp) + '_PPL_wct.png')
    plt.close()
