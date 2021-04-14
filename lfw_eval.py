from PIL import Image
import numpy as np
import bcolz
import os

from sklearn.model_selection import KFold
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
# import torch.backends.cudnn as cudnn

# cudnn.benchmark = True

import net
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# def KFold(n=6000, n_folds=10):
#     folds = []
#     base = list(range(n))
#     for i in range(n_folds):
#         test = base[i * n / n_folds:(i + 1) * n / n_folds]
#         train = list(set(base) - set(test))
#         folds.append([train, test])
#     return folds

multi_sphere = True


def eval_acc(threshold, diff, issame):
    y_predict = np.greater(diff, threshold)
    y_true = np.array(issame)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def get_val_pair(path):
    carray = bcolz.carray(rootdir = os.path.join(path, "lfw"), mode = 'r')
    issame = np.load(os.path.join(path, "lfw_list.npy"))

    return carray, issame

def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

transform = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        F.hflip,
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

def transform_batch(imgs_tensor):
    ccrop_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccrop_imgs[i] = transform(img_ten)

    return ccrop_imgs

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def eval(model_path=None, batch_size=1):
    predicts = []
    model = net.sphere20a().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    root = '/media/data4/zhangzhenduo/dataset/validation dataset'

    # with open('/home/zwp/MyProject/tf_loss/zwp/result/lfw_for_veritification/pairs.txt') as f:
        # pairs_lines = f.readlines()[1:]
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    # ])
    '''
    for i in range(6000):
        p = pairs_lines[i].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

        img1 = Image.open(root + name1).convert('RGB')
        img2 = Image.open(root + name2).convert('RGB')
        img1, img1_, img2, img2_ = transform(img1), transform(F.hflip(img1)), transform(img2), transform(F.hflip(img2))
        img1, img1_ = Variable(img1.unsqueeze(0).cuda(), requires_grad=False), Variable(img1_.unsqueeze(0).cuda(),
                                                                                        requires_grad=False)
        img2, img2_ = Variable(img2.unsqueeze(0).cuda(), requires_grad=False), Variable(img2_.unsqueeze(0).cuda(),
                                                                                        requires_grad=False)
    '''
    carray, issame = get_val_pair(root)
    # print(len(carray))
    # print(len(issame))
    idx = 0
    i = 0
    if multi_sphere:
        embeddings = np.zeros([len(carray), 512//4])
    else:
        embeddings = np.zeros([len(carray), 512])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            # print(batch.size())
            # print(idx)
            transformed = transform_batch(batch)
            transformed = transformed.cuda()
            if multi_sphere:
                embeddings[idx:idx + batch_size] = l2_norm(model(transformed)).cpu()[:, i*512//4:(i+1)*512//4]
            else:
                embeddings[idx:idx + batch_size] = l2_norm(model(transformed)).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            transformed = transform_batch(batch)
            transformed = transformed.cuda()
            if multi_sphere:
                embeddings[idx:] = l2_norm(model(transformed)).cpu()[:, i*512//4:(i+1)*512//4]
            else:
                embeddings[idx:] = l2_norm(model(transformed)).cpu()

    # thresholds = np.arange(0, 4, 0.01)
    f1 = embeddings[0::2]
    f2 = embeddings[1::2]

    # print(np.shape(f1 * f2))
    # print(np.shape((f1 * f2).sum(axis=0)))
    # print(np.shape(np.linalg.norm(f1, axis=0)))
    # print(np.shape(np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0)))
    cosdistance = (f1 * f2).sum(axis=1) / (np.linalg.norm(f1, ord=2, axis=1) * np.linalg.norm(f2, ord=2, axis=1) + 1e-5)

    # cosdistance = torch.sum(torch.matmul(f1, f2), dim=1) / (torch.norm(dim=1) * torch.norm(dim=1) + 1e-5)

    # cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    # predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))
    nrof_pairs = min(len(issame), f1.shape[0])
    indices = np.arange(nrof_pairs)
    accuracy = np.zeros((10))
    best_thresholds = np.zeros((10))
    folds = KFold(n_splits=10, shuffle=False)
    thresholds = np.arange(-1, 1, 0.005)
    nrof_thresholds = len(thresholds)
    # predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))

    f = open("./Test_LFW_log_multi_debug.txt", 'a+')
    
    for fold_idx, (train_set, test_set) in enumerate(folds.split(indices)):
                # Find the best threshold for the fold
        # print(len(train_set))
        # print(len(test_set))
        acc_train = np.zeros((nrof_thresholds))
        # print(train_set.dtype)
        # print(test_set.dtype)
        for threshold_idx, threshold in enumerate(thresholds):
            acc_train[threshold_idx] = eval_acc(threshold, cosdistance[train_set], issame[train_set])
            # print(threshold_idx)
            # print(acc_train[threshold_idx])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        accuracy[fold_idx] = eval_acc(thresholds[best_threshold_index], cosdistance[test_set], issame[test_set])

        # best_thresh = find_best_threshold(thresholds, predicts[train])
        # accuracy.append(eval_acc(best_thresh, predicts[test]))
        # thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(best_thresholds)))
    f.write('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(best_thresholds)) + '\n')
    f.close()

    return np.mean(accuracy)


if __name__ == '__main__':
    '''
    _, result = eval(model_path='checkpoint/SphereFace_24_checkpoint.pth')
    np.savetxt("result.txt", result, '%s')
    '''
    # acc, pred = eval(model_path='./checkpoints/mnface_30_checkpoints.pth')
    # acc, pred = eval(model_path='./YTF_TEST_softmax/checkpoints/softmaxv_26_checkpoints.pth')
    for i in range(30, 1, -1):
        acc, pred = eval(model_path='./YTF_TEST_SHPEREFACE/checkpoints/cosface_'+str(i)+'_checkpoints.pth')
        print('i=' + str(i) + '  acc='+str(acc))
    '''
    for epoch in range(1, 31):
        eval('checkpoint/CosFace_' + str(epoch) + '_checkpoint.pth')
    '''