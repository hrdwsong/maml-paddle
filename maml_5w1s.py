import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import random
import time
from copy import deepcopy, copy
from tqdm import tqdm
import pickle

# 加载训练集和测试集
x_train = np.load('omniglot_train.npy')  # (973, 20, 1, 28, 28)
x_val = np.load('omniglot_val.npy')  # (325, 20, 1, 28, 28)
x_test = np.load('omniglot_test.npy')  # (325, 20, 1, 28, 28)
datasets = {'train': x_train, 'val': x_val, 'test': x_test}

# 全局参数设置
n_way = 5
k_spt = 1  # support data 的个数
k_query = 15  # query data 的个数
imgsz = 28
resize = imgsz
task_num = 32
batch_size = task_num
glob_update_step = 5
glob_update_step_test = 5
glob_meta_lr = 0.001  # 外循环学习率
glob_base_lr = 0.1  # 内循环学习率


indexes = {"train": 0, "val": 0, "test": 0}
print("DB: train", x_train.shape, "validation", x_val.shape, "test", x_test.shape)


def load_data_cache(dataset):
    """
    Collects several batches data for N-shot learning
    :param dataset: [cls_num, 20, 84, 84, 1]
    :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    #  take 5 way 1 shot as example: 5 * 1
    setsz = k_spt * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    for sample in range(50):  # num of epochs

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(batch_size):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], n_way, replace=False)

            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(20, k_spt + k_query, replace=False)

                # 构造support集和query集
                x_spt.append(dataset[cur_class][selected_img[:k_spt]])
                x_qry.append(dataset[cur_class][selected_img[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_spt)
            x_spt = np.array(x_spt).reshape(n_way * k_spt, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_spt)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]

            # append [sptsz, 1, 84, 84] => [batch_size, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        #         print(x_spts[0].shape)
        # [b, setsz = n_way * k_spt, 1, 28, 28]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(np.int64).reshape(batch_size, setsz)
        # [b, qrysz = n_way * k_query, 1, 28, 28]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(np.int64).reshape(batch_size, querysz)
        #         print(x_qrys.shape)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache


datasets_cache = {"train": load_data_cache(x_train),  # current epoch data cached
                  "val": load_data_cache(x_val),
                  "test": load_data_cache(x_test)}


def next(mode='train'):
    """
    Gets next batch from the dataset with name.
    :param mode: The name of the splitting (one of "train", "val", "test")
    :return:
    """
    # 如果所需的index超出当前已经获取的数量，则重新执行load_data_cache获取新的数据
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1

    return next_batch


class MAML(paddle.nn.Layer):
    def __init__(self):
        super(MAML, self).__init__()
        # 定义模型中全部待优化参数
        self.vars = []
        self.vars_bn = []

        # ------------------------第1个conv2d-------------------------
        weight = paddle.static.create_parameter(shape=[64, 1, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        # 第1个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------第2个conv2d------------------------
        weight = paddle.static.create_parameter(shape=[64, 64, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # 第2个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------第3个conv2d------------------------
        weight = paddle.static.create_parameter(shape=[64, 64, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # 第3个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------第4个conv2d------------------------
        weight = paddle.static.create_parameter(shape=[64, 64, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # 第4个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------全连接层------------------------
        weight = paddle.static.create_parameter(shape=[64, n_way],
                                                dtype='float32',
                                                default_initializer=nn.initializer.XavierNormal(),
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[n_way],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])

    def forward(self, x, params=None, bn_training=True):
        """
        :param x: 输入图片
        :param params:
        :param bn_training: set False to not update
        :return: 输出分类
        """
        if params is None:
            params = self.vars

        weight, bias = params[0], params[1]  # 第1个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[2], params[3]  # 第1个BN层
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第1个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第1个MAX_POOL层

        weight, bias = params[4], params[5]  # 第2个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[6], params[7]  # 第2个BN层
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第2个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第2个MAX_POOL层

        weight, bias = params[8], params[9]  # 第3个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[10], params[11]  # 第3个BN层
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第3个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第3个MAX_POOL层

        weight, bias = params[12], params[13]  # 第4个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[14], params[15]  # 第4个BN层
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第4个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第4个MAX_POOL层

        x = paddle.reshape(x, [x.shape[0], -1])  ## flatten
        weight, bias = params[-2], params[-1]  # linear
        x = F.linear(x, weight, bias)

        output = x

        return output

    def parameters(self, include_sublayers=True):
        return self.vars


# # 显示并验证网络结构
# 输入[32,1,28,28]，输出[32,5]
# model = MAML()
#
# x = np.random.randn(*[32, 1, 28, 28]).astype('float32')
# x = paddle.to_tensor(x)
# y = model(x)
# print(y.shape)


class MetaLearner(nn.Layer):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.update_step = glob_update_step  # task-level inner update steps
        self.update_step_test = glob_update_step_test
        self.net = MAML()
        self.meta_lr = glob_meta_lr  # 外循环学习率
        self.base_lr = glob_base_lr  # 内循环学习率
        self.meta_optim = paddle.optimizer.Adam(learning_rate=self.meta_lr, parameters=self.net.parameters())
        # self.meta_optim = paddle.optimizer.Momentum(learning_rate=self.meta_lr,
        #                                             parameters=self.net.parameters(),
        #                                             momentum=0.9)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num = x_spt.shape[0]
        query_size = x_qry.shape[1]  # 75 = 15 * 5
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        # 内循环梯度手动更新，外循环梯度使用定义好的更新器更新
        for i in range(task_num):
            # 第0步更新
            y_hat = self.net(x_spt[i], params=None, bn_training=True)  # (setsz, ways)
            loss = F.cross_entropy(y_hat, y_spt[i])
            grad = paddle.grad(loss, self.net.parameters())  # 计算所有loss相对于参数的梯度和

            tuples = zip(grad, self.net.parameters())  # 将梯度和参数一一对应起来
            # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据，loss填入loss_list_qry[0]，预测正确数填入correct_list[0]
            with paddle.no_grad():
                y_hat = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry
                pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)  # axis取-1也行
                correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
                correct_list[0] += correct
                # 使用更新后的数据在query集上测试。loss填入loss_list_qry[1]，预测正确数填入correct_list[1]
            with paddle.no_grad():
                y_hat = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry
                pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)
                correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
                correct_list[1] += correct

            # 剩余更新步数
            for k in range(1, self.update_step):
                y_hat = self.net(x_spt[i], params=fast_weights, bn_training=True)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = paddle.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

                if k < self.update_step - 1:
                    with paddle.no_grad():
                        y_hat = self.net(x_qry[i], params=fast_weights, bn_training=True)
                        loss_qry = F.cross_entropy(y_hat, y_qry[i])
                        loss_list_qry[k + 1] += loss_qry
                else:  # 对于最后一步update，要记录loss计算的梯度值，便于外循环的梯度传播
                    y_hat = self.net(x_qry[i], params=fast_weights, bn_training=True)
                    loss_qry = F.cross_entropy(y_hat, y_qry[i])
                    loss_list_qry[k + 1] += loss_qry

                with paddle.no_grad():
                    pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)
                    correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
                    correct_list[k + 1] += correct

        loss_qry = loss_list_qry[-1] / task_num  # 计算最后一次loss的平均值
        self.meta_optim.clear_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()

        accs = np.array(correct_list) / (query_size * task_num)  # 计算各更新步数acc的平均值
        loss = np.array(loss_list_qry) / task_num  # 计算各更新步数loss的平均值
        return accs, loss

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        # assert len(x_spt.shape) == 4

        query_size = x_qry.shape[0]
        correct_list = [0 for _ in range(self.update_step_test + 1)]

        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = paddle.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        with paddle.no_grad():
            y_hat = new_net(x_qry, params=new_net.parameters(), bn_training=True)
            pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)
            correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
            correct_list[0] += correct

        # 使用更新后的数据在query集上测试。
        with paddle.no_grad():
            y_hat = new_net(x_qry, params=fast_weights, bn_training=True)
            pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)
            correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
            correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params=fast_weights, bn_training=True)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = paddle.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

            y_hat = new_net(x_qry, fast_weights, bn_training=True)

            with paddle.no_grad():
                pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)
                correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
                correct_list[k + 1] += correct

        del new_net
        accs = np.array(correct_list) / query_size
        return accs


# ------------------------------------------执行训练----------------------------------------
# omniglot
# 设置随机数种子
random.seed(1337)
np.random.seed(1337)

# 开启0号GPU训练
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

meta = MetaLearner()
best_acc = 0
epochs = 10000
print('--------------------{}-way-{}-shot task start!---------------------'.format(n_way, k_spt))
# for step in tqdm(range(epochs)):
for step in range(epochs):
    # start = time.time()
    x_spt, y_spt, x_qry, y_qry = next('train')
    x_spt = paddle.to_tensor(x_spt)
    y_spt = paddle.to_tensor(y_spt)
    x_qry = paddle.to_tensor(x_qry)
    y_qry = paddle.to_tensor(y_qry)
    accs, loss = meta(x_spt, y_spt, x_qry, y_qry)
    # end = time.time()
    if step % 100 == 0:
        print("epoch:", step)
        print(accs)
    #         print(loss)

    if step % 1000 == 0:
        accs = []
        for _ in range(1000 // task_num):
            x_spt, y_spt, x_qry, y_qry = next('val')
            x_spt = paddle.to_tensor(x_spt)
            y_spt = paddle.to_tensor(y_spt)
            x_qry = paddle.to_tensor(x_qry)
            y_qry = paddle.to_tensor(y_qry)

            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)

        print('---------------------在{}个随机任务上测试：---------------------'.format(np.array(accs).shape[0]))
        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('验证集准确率:', accs)
        print('------------------------------------------------------------')
        # 记录并保存最佳模型
        if accs[-1] > best_acc:
            best_acc = accs[-1]
            model_params = [item.numpy() for item in meta.net.vars]
            model_params_file = open('model_param_best_%sway%sshot.pkl' % (n_way, k_spt), 'wb')
            pickle.dump(model_params, model_params_file)
            model_params_file.close()
print('The best acc on validation set is {}'.format(best_acc))

# ------------------------------------------加载模型----------------------------------------
model_params_file = open('model_param_best_%sway%sshot.pkl' % (n_way, k_spt), 'rb')
model_params = pickle.load(model_params_file)
model_params_file.close()
meta = MetaLearner()
meta.net.vars = [paddle.to_tensor(item, stop_gradient=False) for item in model_params]

# ------------------------------------------执行测试----------------------------------------
accs = []
for _ in range(1000 // task_num):
    # db_train.next('test')
    x_spt, y_spt, x_qry, y_qry = next('test')
    x_spt = paddle.to_tensor(x_spt)
    y_spt = paddle.to_tensor(y_spt)
    x_qry = paddle.to_tensor(x_qry)
    y_qry = paddle.to_tensor(y_qry)

    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
        test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
        accs.append(test_acc)

print('---------------------在{}个随机任务上测试：---------------------'.format(np.array(accs).shape[0]))
accs = np.array(accs).mean(axis=0).astype(np.float16)
print('测试集准确率:', accs)
print('------------------------------------------------------------')
