import numpy

# 这个是纯加载数据，分别data和label是拆开的
# 这里cifar一共50000个训练样本，每个是3072，需要reshape成3*32*32
#我们不采用torch的方式，因为原本代码的缘故，我们需要拆分数据和标签
def load_cifar10(filename):
    import pickle
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
    x=dataset[b'data']
    y=dataset[b'labels']
    return x,y

#cifar10只有至多50000个训练样本，没有验证数据集
def dataload_cifar10():
    import pickle
    filename_train1 = './data/Cifar/Cifar-10-batches-py/data_batch_1'
    filename_train2 = './data/Cifar/Cifar-10-batches-py/data_batch_2'
    filename_train3 = './data/Cifar/Cifar-10-batches-py/data_batch_3'
    filename_train4 = './data/Cifar/Cifar-10-batches-py/data_batch_4'
    filename_train5 = './data/Cifar/Cifar-10-batches-py/data_batch_5'
    filename_test = './data/Cifar/Cifar-10-batches-py/test_batch'

    x_train1, y_train1 = load_cifar10(filename_train1)
    x_train2, y_train2 = load_cifar10(filename_train2)
    x_train3, y_train3 = load_cifar10(filename_train3)
    x_train4, y_train4 = load_cifar10(filename_train4)
    x_train5, y_train5 = load_cifar10(filename_train5)

    x_train = numpy.concatenate((x_train1, x_train2, x_train3, x_train4, x_train5), 0)
    y_train = numpy.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5), 0)

    x_test, y_test = load_cifar10(filename_test)
    x_valid, y_valid = load_cifar10(filename_test)

    x_train = x_train.reshape(50000, 3, 32, 32)
    x_test = x_test.reshape(10000, 3, 32, 32)
    x_valid = x_valid.reshape(10000, 3, 32, 32)

    return x_train, y_train, x_valid, y_valid, x_test, y_test