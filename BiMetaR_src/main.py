from trainer import *
from params import *
from data_loader import *
import pickle as pl
import json
from tqdm import tqdm
from collections import defaultdict


if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = ''
    if params['data_form'] == 'In-Train':
        tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks'+tail]))
    # print("loading test_tasks ... ...")
    # dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print('loading test support set ... ...')
    dataset['test_tasks'] = json.load(open(data_dir['test_support']))
    print('loading test query set ... ...')
    dataset['test_query'] = json.load(open(data_dir['test_query']))
    print('loading groundtruth ... ...')
    dataset['test_groundtruth'] = []
    with open(data_dir['test_gt'], "r") as f:
        reader = csv.reader(f, delimiter=',')
        for id, line in enumerate(reader):
            if id==0:
                continue
            dataset['test_groundtruth'].append(line[1])
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates'+tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2'+tail]))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    print("loading relation2id ... ...")
    dataset['rel2id'] = json.load(open(data_dir['relation2ids']))

    # if params['data_form'] == 'Pre-Train':
    print('loading embedding ... ...')
        # with open(data_dir['ent2vec'], 'rb') as f:
        #     dataset['ent2emb'] = pl.load(f)
        
    dataset['ent2emb'] = np.loadtxt(open(data_dir['ent2vec'], "r"), delimiter=",", skiprows=0)
    dataset['rel2emb'] = np.loadtxt(open(data_dir['rel2vec'], "r"), delimiter=",", skiprows=0)
        # print(data.shape)   #(68543, 100)
    dataset['path_graph'] = data_dir['path_graph']
    
    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        # trainer.reload()
        # trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        # if params['eval_by_rel']:
        #     trainer.eval_by_relation(istest=True)
        # else:
        #     trainer.eval(istest=True)
        # trainer.reload()
        trainer.eval(istest=True)

    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)

