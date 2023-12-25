# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger
import datetime
import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
# jiakai
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import manifold
import numpy as np

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def visual_embedding_tsne(log_dir,config,all_embeddings_list, title, sample_num=3000):

    def get_sample_embeddings(all_embeddings):
        if sample_num > all_embeddings.shape[0]:
            raise ValueError('Sample Num > Embedding Num!')
        sample_inds = np.random.choice(all_embeddings.shape[0], size=sample_num, replace=False)
        sample_embeddings = all_embeddings[sample_inds]
        return sample_embeddings

    def plot_kde_xy(df, model_name, ax, show_ylabel=True):
        # sns.kdeplot(data=df, x='x', y='y', thresh=0., level=100, cmap=sns.color_palette('light:g', 8, as_cmap=True),
        #             ax=ax,shade=True)
        sns.kdeplot(data=df, x='x', y='y', thresh=0., level=100, cmap=sns.color_palette('light:g',8, as_cmap=True),ax=ax,shade=True)
        # 'light:g' 'hls',8 1 'Paired',10   shade=True  as_cmap=True level=100
        ax.set_xlabel('Features', fontsize=12)
        if show_ylabel:
            ax.set_ylabel('Features', fontsize=12)
        else:
            ax.set_ylabel('Features', fontsize=12)
        # ax.set_title(model_name, fontsize=12, fontweight="bold")

    def plot_kde_angles(angles, ax):
        # sns.histplot(data=angles, ax=axs[1][0], color='green', kde=True, edgecolor='none')
        sns.kdeplot(data=angles, ax=ax, color='green', shade=True, linewidth=2) # green blue purple pink green
        ax.set_xlabel('Angles', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, 0.3)

    # model_names = ['RGCL_LGC[1]', 'RGCL_LGC[2]', 'RGCL_LGC[3]', 'RGCL_LGC[4]', 'RGCL_LGC[5]']
    model_names = ['DuoRec']
    model_num = len(all_embeddings_list)
    fig, axs = plt.subplots(2, model_num, gridspec_kw={'height_ratios': [5, 2]})
    fig.set_size_inches(3 * len(model_names), 6)
    # fig.suptitle(title, fontsize=16, fontweight='bold')

    x=all_embeddings_list[0]
    sample_embeddings = get_sample_embeddings(x)
    tsne = manifold.TSNE(n_components=2, init='pca',random_state=0)
    tsne_sample_embeddings = tsne.fit_transform(sample_embeddings)
    norm_sample_embeddings = normalize(tsne_sample_embeddings)
    #sample_embeddings_list.append(norm_sample_embeddings)
    sample_embeddings_list =norm_sample_embeddings

    #for i, (embs, name) in enumerate(zip(sample_embeddings_list, model_names)):
    embs = sample_embeddings_list
    name =model_names[0]
    df = pd.DataFrame({'x': embs.T[0],'y': embs.T[1]})
    plot_kde_xy(df, name, axs[0])
    # plot_kde_xy(df, axs[0])
    x = embs.T[0]
    y = embs.T[1]
    angles = np.arctan2(y, x)
        # plot_kde_angles(angles, axs[1][i])
    plot_kde_angles(angles, axs[1])

    # axs[1][0].set_ylim(0, 0.2)
    plt.tight_layout()
    # plt.savefig(log_dir + '/' + config['model'] + '-' + config['dataset'] + '-' + 'contra_norm' + '.pdf', format='pdf', transparent=False, bbox_inches='tight')
    plt.savefig(log_dir + '/' + config['model'] + '/' + config['dataset'] + '/' + '-' + config['Oversmoothing']+ '-' +'tau'+ str(config['Tau'])+ '-' +'Scale'+ str(config['Scale']) + config['nowtime'] + '-' +'contra_norm' + '.pdf', format='pdf',
                transparent=False, bbox_inches='tight')

    plt.show()

def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    import os
    # log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    # config['log_dir'] = log_dir
    # config['log_dir'] = 'log'
    config['nowtime'] = get_local_time()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    embedding_matrix = model.item_embedding.weight[1:].cpu().detach().numpy()
    item_embeddings_list = [embedding_matrix]
    # visual_embedding_tsne(args,item_embeddings_list, 'Item 2000', 2000)
    visual_embedding_tsne('log', config, item_embeddings_list, 'Item', item_embeddings_list[0].shape[0])

    svd = TruncatedSVD(n_components=2)
    svd.fit(embedding_matrix)
    comp_tr = np.transpose(svd.components_)
    proj = np.dot(embedding_matrix, comp_tr)

    cnt = {}
    for i in dataset['item_id']:
        if i.item() in cnt:
            cnt[i.item()] += 1
        else:
            cnt[i.item()] = 1

    freq = np.zeros(embedding_matrix.shape[0])
    for i in cnt:
        freq[i - 1] = cnt[i]

    # freq /= freq.max()

    #sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.scatter(proj[:, 0], proj[:, 1], s=3, c=freq, cmap='viridis_r') # viridis_r OrRd viridis
    plt.colorbar()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    # plt.axis('square')
    # plt.show()
    # print("config['model']",config['model'])
    # exit()
    # plt.savefig('log', + '/' + config['model'] + '-' + config['dataset'] + '.pdf', format='pdf', transparent=False,
    #             bbox_inches='tight')
    plt.savefig('log' + '/' + config['model'] + '/' + config['dataset'] + '/' + config['Oversmoothing'] + '-' +'tau'+ str(config['Tau'])+ '-' +'Scale'+ str(config['Scale'])+'n_layers'+ '-'+ str(config['n_layers']) + '-'+ config['nowtime'] + '-'+ '2dlayer'  + '.pdf', format='pdf',
                transparent=False, bbox_inches='tight')

    from scipy.linalg import svdvals
    embedding_matrix = normalize(embedding_matrix)
    svs = svdvals(embedding_matrix)
    svs = np.log(svs)
    # svs /= svs.max()
    # np.save('log', + '/sv.npy', svs)
    # plt.savefig('log' + '/' + config['model'] + '/' + config['dataset'] + '/' + '-' + 'svs-picture' + '.pdf', format='pdf',
    #             transparent=False, bbox_inches='tight')
    sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.plot(svs)
    # plt.show()
    # plt.savefig('log', + '/svs.pdf', format='pdf', transparent=False, bbox_inches='tight')
    plt.savefig('log' + '/' + config['model'] + '/' + config['dataset'] + '/'  + config['Oversmoothing']+ '-' +'tau'+ str(config['Tau'])+ '-' +'Scale'+ str(config['Scale'])  + config['nowtime'] + '-' + 'svs-singular' + '.pdf', format='pdf',
                transparent=False, bbox_inches='tight')

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
