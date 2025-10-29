import glob
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def beautify_experiment_name(name):
    if "transfer_False-restricted_False" in name:
        return "No Transfer"
    elif "transfer_True-restricted_False" in name: 
        return "Transfer All"
    elif "transfer_True-restricted_True" in name:
        return "Transfer Neighborhood"
    else:
        return 'Unknown Experiment'

def plot_reward(data, chart_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='step', y='MeanTeamReward', hue='experiment', linewidth=2.5)
    plt.xlabel('Step')
    plt.ylabel('Mean Team Reward')
    plt.legend(title='Experiment')
    plt.tight_layout()
    plt.savefig(f'{chart_path}/results-reward.pdf')

def plot_loss(data, chart_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='step', y='MeanLoss', hue='experiment', linewidth=2.5)
    plt.xlabel('Step')
    plt.ylabel('Mean Loss')
    plt.legend(title='Experiment')
    plt.tight_layout()
    exp = data['experiment'][0]
    plt.savefig(f'{chart_path}/results-loss-{exp}.pdf')

if __name__ == '__main__':

    experiments = [
        "SimpleSpread-transfer_False-restricted_False",
        "SimpleSpread-transfer_True-restricted_False",
        "SimpleSpread-transfer_True-restricted_True"
    ]

    data_path = 'data'
    chart_path = 'charts/'
    Path(chart_path).mkdir(parents=True, exist_ok=True)

    data_train = []
    data_eval = []

    for experiment in experiments:
        # TODO - this will be useful when with more seeds
        #df_train = glob.glob(f'{data_path}/{experiment}/train-results-seed_0.csv')
        #df_eval = glob.glob(f'{data_path}/{experiment}/eval-results-seed_0.csv')
        df_train = pd.read_csv(f'{data_path}/{experiment}/train-results-seed_0.csv')
        df_eval = pd.read_csv(f'{data_path}/{experiment}/eval-results-seed_0.csv')
        exp = beautify_experiment_name(experiment)
        df_train['experiment'] = exp
        df_train['step'] = df_train.index
        df_eval['experiment'] = exp
        df_eval['step'] = df_eval.index
        data_train.append(df_train)
        data_eval.append(df_eval)

    #df_train = pd.concat(data_train, ignore_index=True)
    df_eval = pd.concat(data_eval, ignore_index=True)
    for df in data_train:
        plot_loss(df, chart_path)
    plot_reward(df_eval, chart_path)
