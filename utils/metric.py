import statistics

from loguru import logger
import numpy as np
from requests import session
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from collections import defaultdict


class Metric:
    """
    best_subject_avg_acc: the best average accuracy across all sessions for each subject
    best_two_sessions_subject_avg_acc: the best average accuracy across best two 
        sessions for each subject
    best_one_session_subject_avg_acc: the best average accuracy across best one 
        session for each subject
    subject_all_sessions_acc: the list of accuracy for each session of each subject
    """
    def __init__(self,num_subjects:int, num_session:int):
        self.accuracy  = np.zeros((num_subjects, num_session))

    def update(self, subject_id:int,session_id:int, acc:float):
        if acc > self.accuracy[subject_id, session_id]:
            logger.info("--> A new better acc {:<.4f} on subject {} session {}", acc, subject_id, session_id)
            self.accuracy[subject_id, session_id] = acc

    def all_sessions_mean_acc(self)->tuple[float, float]:
        """
        return
            The mean of acc on all subjects and all sessions
            The std of acc on all subjects and all sessions
        """
        return self.accuracy.mean(), self.accuracy.std()
    
    def two_best_sessions_mean_acc(self)->tuple[float, float]:
        """
        return
            The mean of acc on all subjects and two best sessions
            The std of acc on all subjects and two best sessions
        """
        np.sort(self.accuracy, axis=1)
        return self.accuracy[:, :2].mean(), self.accuracy[:, :2].std()
    
    def one_best_session_mean_acc(self)->tuple[float, float]:
        """
        return
            The mean of acc on all subjects and one best session
            The std of acc on all subjects and one best session
        """
        return self.accuracy.max(axis=1).mean(), self.accuracy.max(axis=1).std()


class Metric_:
    """
    Use the values class to calculate various metrics
    """
    def __init__(self, metrics):
        # record the output values and target values of the model for each batch
        self.values = {}
        self.outputs = []
        self.targets = []
        self.losses = []
        self.metrics = metrics

    def accuracy(self):
        self.values['acc'] = accuracy_score(self.targets,self.outputs)
        # calculate the accuracy
        return self.values['acc']

    def update(self, outputs, targets, loss=None):
        # append one batch outputs and targets to all outputs and targets
        if torch.is_tensor(outputs):
            self.outputs += outputs.cpu().detach().tolist()
            self.targets += targets.cpu().detach().tolist()
        else:
            self.outputs += outputs.tolist()
            self.targets += targets.tolist()
        if loss is not None:
            self.losses.append(loss)

    def macro_f1_score(self):
        self.values['macro-f1'] = f1_score(self.targets, self.outputs, average='macro')
        # calculate the macro f1-score
        return self.values['macro-f1']

    def micro_f1_score(self):
        self.values['micro-f1'] = f1_score(self.targets, self.outputs, average='micro')
        # calculate the micro f1-score
        return self.values['micro-f1']

    def weighted_f1_score(self):
        self.values['weighted-f1'] = f1_score(self.targets, self.outputs, average='weighted')
        return self.values['weighted-f1']

    def ck_coe(self):
        self.values['ck'] = cohen_kappa_score(self.targets, self.outputs)
        # calculate the micro f1-score
        return self.values['ck']
    def value(self):
        # 　if one hot code, then transform to ordinary label
        if type(self.targets[0]) is list:
            try:
                self.targets = [self.targets[i].index(1) for i in range(len(self.targets))]
            except ValueError:
                return "unavailable"
        func = {
            'acc': self.accuracy,
            'macro-f1': self.macro_f1_score,
            'micro-f1': self.micro_f1_score,
            'ck': self.ck_coe,
            'weighted-f1': self.weighted_f1_score,
        }
        out = ""
        for m in self.metrics:
            out += f"{m}: {func[m]():.3f}   "
        if len(self.losses) != 0:
            return out + f"loss: {sum(self.losses)/len(self.losses):.4f}"
        else:
            return out
            
class SubMetric(Metric):
    def __init__(self,metrics):
        super().__init__(metrics=metrics)
        self.sub_labels = []
    def update(self, outputs, targets, sub_labels, loss=None):
        # append one batch outputs and targets to all outputs and targets
        if torch.is_tensor(outputs):
            self.outputs += outputs.cpu().detach().tolist()
            self.targets += targets.cpu().detach().tolist()
            self.sub_labels += sub_labels.cpu().detach().tolist()
        else:
            self.outputs += outputs.tolist()
            self.targets += targets.tolist()
            self.sub_labels += sub_labels.cpu().detach().tolist()
        if loss is not None:
            self.losses.append(loss)
    def sub_accuracy(self):
        # this function will return every subject's acc
        groups = defaultdict(lambda : {"outputs":[], "targets":[]})
        for o, t, s in zip(self.outputs, self.targets, self.sub_labels):
            groups[s]["outputs"].append(o)
            groups[s]["targets"].append(t)
        result = {}
        for label in groups:
            acc = accuracy_score(groups[label]["targets"], groups[label]["outputs"])
            result[label] = acc
        return result
    def sub_macro_f1_score(self):
        # this function will return every subject's macro-f1
        groups = defaultdict(lambda: {"outputs": [], "targets": []})
        for o, t, s in zip(self.outputs, self.targets, self.sub_labels):
            groups[s]["outputs"].append(o)
            groups[s]["targets"].append(t)
        result = {}
        for label in groups:
            acc = f1_score(groups[label]["targets"], groups[label]["outputs"], average='macro')
            result[label] = acc
        return result

    def accuracy(self):
        """
        this function will return the mean and std acc of all subjects
        :return:
        """
        acc_dict =  list(self.sub_accuracy().values())
        self.values['acc'] = statistics.mean(acc_dict)
        if len(acc_dict) != 1:
            self.values['acc_std'] = statistics.stdev(acc_dict)
        return self.values['acc']

    def macro_f1_score(self):
        """
        this function will return the mean and std macro-f1 of all subjects
        :return:
        """
        macro_f1_s = list(self.sub_macro_f1_score().values())
        self.values['macro-f1'] = np.mean(macro_f1_s)
        if len(macro_f1_s) != 1:
            self.values['macro-f1_std'] =  np.std(macro_f1_s)
        return self.values['macro-f1']