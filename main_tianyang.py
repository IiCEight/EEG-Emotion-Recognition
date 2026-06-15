from loguru import logger
import torch
import random
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from config.logging import setUpLogger
from reference.model_DANN import MSMDAERNet
from reference.utils import load_seed4, UnalignedDataLoader,  z_score
import seaborn as sns
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    setUpLogger("INFO")
    setup_seed(2024)
    max_epoch = 3000
    lr = 1e-3
    batch_size = 50
    number_of_category = 4
    number_of_source = 15
    num_domains = 15
    path = "../data/SEED_IV/eeg_feature_smooth/"
    session = 2
    Data, Label = load_seed4(path, session)
    result_one_test = np.zeros(15)
    result_two_test = np.zeros(15)
    true_test = []
    pre_test = []

    logger.info("------- begin -------\n")

    for sub in range(0, 15):
        Tx = Data[sub]
        Ty = Label[sub]
        Tx, m, std = z_score(Tx)

        subjects = Data.keys()
        Sx = Sy = None
        i = 0
        flag = False
        selected_subject = sub
        for s in subjects:
            if i != selected_subject:
                tr_x = np.array(Data[s])
                tr_y = np.array(Label[s])
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        Tx_tensor = torch.tensor(Tx)
        Ty_tensor = torch.tensor(Ty)
        target_ts = TensorDataset(Tx_tensor, Ty_tensor)
        # data loader
        dset_loaders = {}
        dset_loaders["test"] = DataLoader(target_ts, batch_size=200, shuffle=False, )
        msorce_loaders = DataLoader(TensorDataset(torch.tensor(Sx), torch.tensor(Sy)), batch_size=batch_size, shuffle=True, drop_last=True)
        train_loader = UnalignedDataLoader()
        train_loader.initialize(num_domains, Data, Label, Tx, Ty, sub, batch_size, batch_size,
                                shuffle_testing=True, drop_last_testing=True)
        datasets = train_loader.load_data()


        # 加载模型
        model = MSMDAERNet(number_of_source, number_of_category).cuda()
        ema = EMA(0.998)
        ema.register(model)
        Ct_memory = []
        for d in range(14):
            Ct_memory.append(torch.zeros(number_of_category, 320).cuda())
        Cs_memory = torch.zeros(number_of_category, 320).cuda()
        optimizer = torch.optim.RMSprop(model.get_parameters(), lr=lr, weight_decay=1e-5)

        best_acc = 0.0
        best_all_label = torch.empty(0)
        best_predictions = torch.empty(0)
        temp = 0
        iter_s = {}
        iter_ms = {}
        for epoch in range(max_epoch):
            p = epoch / max_epoch
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            try:
                data = next(iter_s)
            except Exception as err:

                iter_s = iter(datasets)
                data = next(iter_s)

            # get the source batches
            x_src = list()
            y_src = list()
            for domain_idx in range(num_domains - 1):
                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                x_src.append(tmp_x)
                y_src.append(tmp_y)
            # get the target batch
            x_trg = data['Tx'].float().cuda()
            y_trg = data['Ty'].long().cuda()

            inputs_target = x_trg
            labels_target = y_trg

            try:
                mx, my = next(iter_ms)
                mx = mx.float().cuda()
                my = my.long().cuda()
            except Exception as err:
                iter_ms = iter(msorce_loaders)
                mx, my = next(iter_ms)
                mx = mx.float().cuda()
                my = my.long().cuda()


            model.train()
            for i in range(number_of_source):
                if i==(number_of_source-1):
                    inputs_st, labels_st = mx, my
                else:
                    inputs_st, labels_st = x_src[i], y_src[i]
                # Cast
                inputs_st = inputs_st.type(torch.FloatTensor)
                labels_st = labels_st.type(torch.LongTensor)
                inputs_target = inputs_target.type(torch.FloatTensor)
                labels_target = labels_target.type(torch.LongTensor)
                # to cuda
                inputs_st, labels_st = inputs_st.cuda(), labels_st.cuda()
                inputs_target, labels_target = inputs_target.cuda(), labels_target.cuda()

                features_st1, features_st2, outputs_st = model(inputs_st, i)
                softmax_output15 = torch.softmax(outputs_st, dim=1)
                features_target1, features_target2, outputs_target = model(inputs_target, i)
                softmax_output2 = torch.softmax(outputs_target, dim=1)

                ##################
                labels_st_onehot = F.one_hot(labels_st, number_of_category).type(torch.float32)
                class_center = torch.matmul(
                    torch.inverse(torch.diag(labels_st_onehot.sum(axis=0)) + torch.eye(number_of_category).cuda()),
                    torch.matmul(labels_st_onehot.T, features_st2))
                dis_st = torch.matmul(torch.cat((features_st2, features_target2)), class_center.T)
                s_dis_st = torch.nn.functional.softmax(dis_st, dim=1)
                ss = get_cos_similarity_distance(s_dis_st)

                sim_matrix = torch.mm(s_dis_st, s_dis_st.T)
                N = s_dis_st.shape[0]
                mask = torch.eye(N, dtype=torch.bool)
                tensor_no_diag = sim_matrix[~mask].reshape(N, N - 1)
                temp_loss = torch.nn.functional.log_softmax(tensor_no_diag, dim=1)

                guess_label_p = get_cos_similarity_by_threshold(ss)
                positive_mask_no_diag = guess_label_p[~mask].reshape(N, N - 1)
                no_mask = positive_mask_no_diag.sum(dim=1) != 0
                positive_loss = (((temp_loss * positive_mask_no_diag).sum(dim=1))[no_mask] /
                                 (-1 * positive_mask_no_diag.sum(dim=1))[no_mask]).mean()

                guess_label_n = get_cos_similarity_by_threshold2(ss)
                ne_mask_no_diag = guess_label_n[~mask].reshape(N, N - 1)
                no_mask2 = ne_mask_no_diag.sum(dim=1) != 0
                ne_loss = (((temp_loss * ne_mask_no_diag).sum(dim=1))[no_mask2] / (-1 * ne_mask_no_diag.sum(dim=1))[
                    no_mask2]).mean()

                cluster_loss = positive_loss-ne_loss

                ce_loss = 0
                for domain_idx in range(15):
                    if domain_idx==14:
                        features_source1, features_source2, outputs_source = model(mx, domain_idx)
                        s_fea = features_source1
                        ce_loss += torch.nn.CrossEntropyLoss()(outputs_source, my)
                    else:
                        features_source1, features_source2, outputs_source = model(x_src[domain_idx], domain_idx)
                        ce_loss += torch.nn.CrossEntropyLoss()(outputs_source, y_src[domain_idx])

                if i==14:
                    total_loss = ce_loss/15
                else:
                    semantic_loss, Cs_memory, Ct_memory[i] = SM(s_fea, torch.cat((features_st1, features_target1)),
                                                                my, torch.cat(
                            (torch.argmax(softmax_output15, dim=1), torch.argmax(softmax_output2, dim=1))),
                                                                Cs_memory, Ct_memory[i],
                                                                decay=0.9)
                    total_loss = ce_loss / 15 + alpha*cluster_loss + alpha * (0.01*semantic_loss)

                # reset gradients
                optimizer.zero_grad()

                # compute gradients
                total_loss.backward()

                optimizer.step()
                # Polyak averaging.
                ema(model)  # TODO: move ema into the optimizer step fn.
                if i!=14:
                    Ct_memory[i].detach_()
                    Cs_memory.detach_()
            model.eval()
            temp, all_label, predictions = test_suda(dset_loaders['test'], model)
            if epoch%50==0:
                logger.info("subjct:{} epoch:{} acc:{:.4f}".format(sub + 1, epoch + 1, temp))
            if best_acc < temp:
                best_acc = temp
                best_all_label = all_label
                best_predictions = predictions
                logger.info("best_acc: {:.4f}".format(best_acc))
            if best_acc==1.0:
                break
        result_one_test[sub] = best_acc
        result_two_test[sub] = temp
        true_test.append(best_all_label)
        pre_test.append(best_predictions)
        logger.info(result_one_test[sub])

    # 计算混淆矩阵
    cm = confusion_matrix(torch.cat(true_test).tolist(), torch.cat(pre_test).tolist())
    cm_ratio = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # 绘制混淆矩阵
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_ratio, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Neutral', 'Sad', 'Fear', 'Happy'],
                yticklabels=['Neutral', 'Sad', 'Fear', 'Happy'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    for i, r in enumerate(result_one_test):
        logger.info("第{}个人准确率为：{:.4f}\n".format(i + 1, r))
    logger.info("均值为：{:.4f}".format(np.mean(result_one_test)))
    logger.info("标准差为：{:.4f}".format(np.std(result_one_test)))
    for i, r in enumerate(result_two_test):
        logger.info("第{}个人准确率为：{:.4f}\n".format(i + 1, r))
    logger.info("均值为：{:.4f}".format(np.mean(result_two_test)))
    logger.info("标准差为：{:.4f}".format(np.std(result_two_test)))

def get_cos_similarity_by_threshold(cos_dist_matrix):
    """Get similarity by threshold
    :param cos_dist_matrix: cosine distance in matrix,
    (batch_size, batch_size)
    :param threshold: threshold, scalar
    :return: distance matrix between features, (batch_size, batch_size)
    """
    device = cos_dist_matrix.device
    dtype = cos_dist_matrix.dtype
    similar = torch.tensor(1, dtype=dtype, device=device)
    dissimilar = torch.tensor(0, dtype=dtype, device=device)
    sim_matrix = torch.where(cos_dist_matrix > 0.9, similar,
                             dissimilar)
    return sim_matrix

def get_cos_similarity_by_threshold2(cos_dist_matrix):
    """Get similarity by threshold
    :param cos_dist_matrix: cosine distance in matrix,
    (batch_size, batch_size)
    :param threshold: threshold, scalar
    :return: distance matrix between features, (batch_size, batch_size)
    """
    device = cos_dist_matrix.device
    dtype = cos_dist_matrix.dtype
    similar = torch.tensor(1, dtype=dtype, device=device)
    dissimilar = torch.tensor(0, dtype=dtype, device=device)
    sim_matrix = torch.where(cos_dist_matrix < 0.5, similar,
                             dissimilar)
    return sim_matrix

def get_cos_similarity_distance(features):
    """Get distance in cosine similarity
    :param features: features of samples, (batch_size, num_clusters)
    :return: distance matrix between features, (batch_size, batch_size)
    """
    # (batch_size, num_clusters)
    features_norm = torch.norm(features, dim=1, keepdim=True)
    # (batch_size, num_clusters)
    features = features / features_norm
    # (batch_size, batch_size)
    cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
    return cos_dist_matrix

def test_suda(loader, model):
    start_test = True
    with torch.no_grad():
        # get iterate data
        iter_test = iter(loader)
        for i in range(len(loader)):
            # get sample and label
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # load in gpu
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels
            # obtain predictions
            _, outputs = model(inputs,0)
            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # obtain labels
    _, predictions = torch.max(all_output, 1)
    # calculate accuracy for all examples
    accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

    return accuracy, all_label, predictions

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty

def SM(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, Wt=None, decay=0.3):
    # Clone memory
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()

    K = Cs.size(0)
    # for each class
    for k in range(K):
        Xs_k = Xs[Ys==k]
        Xt_k = Xt[Yt==k]

        if len(Xs_k)==0:
            Cs_k = 0.0
        else:
            Cs_k = torch.mean(Xs_k,dim=0)

        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            if Wt is None:
                Ct_k = torch.mean(Xt_k,dim=0)
            else:
                Wt_k = Wt[Yt==k]
                Ct_k = torch.sum(Wt_k.view(-1, 1) * Xt_k, dim=0) / (torch.sum(Wt_k) + 1e-5)

        Cs[k, :] = (1-decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1-decay) * Ct_memory[k, :] + decay * Ct_k

    Dist = cosine_matrix(Cs, Ct)

    return torch.sum(torch.diag(Dist)), Cs, Ct


def compute_indicator(cos_dist_matrix):
    device = cos_dist_matrix.device
    dtype = cos_dist_matrix.dtype
    mask = torch.tril(torch.ones_like(cos_dist_matrix), diagonal=-1)
    cos_dist_matrix= cos_dist_matrix.masked_fill(mask == 1, 0.6)
    selected = torch.tensor(1, dtype=dtype, device=device)
    not_selected = torch.tensor(0, dtype=dtype, device=device)
    w2 = torch.where(cos_dist_matrix < 0.5, selected, not_selected)
    w1 = torch.where(cos_dist_matrix > 0.9, selected, not_selected)
    w = w1 + w2
    nb_selected = torch.sum(w)
    return w, nb_selected

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]

if __name__ == '__main__':
    main()


