import torch
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score 
from prettytable import PrettyTable


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from sklearn.model_selection import LeaveOneGroupOut
from natsort import natsorted
import argparse
import time
import mlflow
from sklearn.naive_bayes import CategoricalNB

from samm_dataset import *
from samm_models import TinyVIT, res18_imagenet, CultureAwareClassifier, CulturalMTL, CulturalDualNetwork_LateFusion

# from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, NormalDistribution, Node, State
from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import Categorical, Bernoulli, Normal
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayes_classifier import BayesClassifier


from IPython.display import display, Math, Latex,HTML
import pyagrum as gum
import pyagrum.lib.notebook as gnb
import pyagrum.causal as csl
import pyagrum.causal.notebook as cslnb
import pyagrum.lib.discretizer as disc
import pyagrum as gum



def test_samm(args):
    root_dir = "/home/hq/Documents/data/SAMM/SAMM_CROP"
    # annotation_file = "/home/hq/Documents/WorkingRepos/AU_Localization/CD6ME_Ethnic/JointDB_MetaEmotionConcised.xlsx"

    # root_dir = '/home/hq/Documents/data/'
    # annotation_file = "/home/hq/Documents/WorkingRepos/AU_Localization/CD6ME_Ethnic/JointDB_MetaEmotionConcised.xlsx"
    annotations_casme2 = pd.read_excel('/home/hq/Documents/WorkingRepos/AU_Localization/CD6ME_Ethnic/CASME2MetaEmotionConcised.xlsx')
    annotations_samm = pd.read_excel('/home/hq/Documents/WorkingRepos/AU_Localization/CD6ME_Ethnic/SAMMMetaEmotionConcised.xlsx')

    annotations_samm['Subject'] = annotations_samm['Subject'].astype(str).str.zfill(3)
    annotations_casme2['Subject'] = annotations_casme2['Subject'].astype(str).str.zfill(2)
    annotations_casme2['Subject'] = annotations_casme2['Subject'].apply(lambda x: 'sub' + x if len(x) == 2 else x)    
    annotations_casme2['Dataset'] = 'CASME2'
    annotations_samm['Dataset'] = 'SAMM'

    annotation_file = pd.concat([annotations_casme2, annotations_samm], ignore_index=True)
    

    
    weights_path = args.weights_path
    weights_name = args.weights_name

    batch_size = args.batch_size
    epochs = args.epochs



    weights_path = os.path.join(weights_path, weights_name)

    # Get all LOSO splits
    # sampled_asian_df, sampled_non_asian_df, sampled_mixed_df = racial_based_sampling(annotation_file, concised_race=False)
    # splits = get_loso_splits(sampled_asian_df)
    # splits = get_loso_splits(sampled_non_asian_df)
    # splits = get_loso_splits(sampled_mixed_df)
    # splits = get_loso_splits(annotation_file)

    # Get all LOSO splits
    splits = get_loso_splits(annotation_file)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), antialias=True),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    optical_flow_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),  # Resize to model input size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    table = PrettyTable()
    # table.field_names = ['Subject', 'loss', 'Fear', 'Other', 'Surprise', 'Happiness', 'Anger', 'Disgust', 'Contempt', 'Sadness', 'avg_f1']
    # table.field_names = ['Subject', 'loss', 'Other', 'Surprise', 'Happiness', 'Anger', 'Contempt', 'avg_f1']
    table.field_names = ['Subject', 'loss', 'Positive', 'Negative', 'Surprise', 'avg_f1']
    softmax = nn.Softmax(dim=1)

    all_folds_predictions_list = []
    all_folds_labels_list = []   
    all_folds_race_list = []

    all_folds_ace = []
    all_folds_mape = []

    for subject_out, (train_data, test_data) in splits.items():
        
        print(f"Leaving out subject: {subject_out}")

        # Train and Test datasets
        test_dataset = JointDBDataset(root_dir, annotation_file, train_test_flag='test', subject_out=subject_out, transform=transform, of_transform=optical_flow_transform)

        # Data Loaders
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # model
        # loading weights
        weights_str = os.path.join(weights_path, weights_name)
        if os.path.exists(weights_str) == False:
            os.makedirs(weights_str)
        weights_str = "{}_{}.pth".format(weights_str, subject_out)
        weights = torch.load(weights_str)
        # model = res18_imagenet(num_classes=3).cuda()
        model = CulturalDualNetwork_LateFusion(num_classes=3, num_cultures=2).cuda()   
        model.load_state_dict(weights)
        model.eval()
        
        with torch.no_grad():

            for batch in test_loader:
                frames, of_frames, labels, race = batch
                frames = frames.cuda()
                of_frames = of_frames.float().cuda()
                # of_frames = of_frames.permute(0, 3, 1, 2)
                labels = labels.cuda()      
                labels = labels.to(torch.int64)
                race = race.to(torch.int64).cuda()
                # gender = gender.cuda()

                # predicted_emotions = model(of_frames)
                emote_logits, race_logits, predicted_emotions = model(of_frames, frames)
                predicted_emotions = softmax(predicted_emotions)
                race_logits = softmax(race_logits)
                predicted_emotions = torch.argmax(predicted_emotions, 1)
                predicted_race =  torch.argmax(race_logits, 1)
                all_folds_predictions_list += [predicted_emotions.cpu().numpy()]
                all_folds_labels_list += [labels.cpu().numpy()]     
                all_folds_race_list += [race.cpu().numpy()]
                # all_folds_predictions_list += [predicted_emotions.cpu().numpy()]




    table = PrettyTable()
    # table.field_names = ['Fear', 'Other', 'Surprise', 'Happiness', 'Anger', 'Disgust', 'Contempt', 'Sadness', 'avg_f1']
    # table.field_names = ['Other', 'Surprise', 'Happiness', 'Anger', 'Contempt', 'avg_f1']
    table.field_names = ['Positive', 'Negative', 'Surprise', 'avg_f1']
    predictions_list = np.concatenate(all_folds_predictions_list)
    labels_list = np.concatenate(all_folds_labels_list)
    race_list = np.concatenate(all_folds_race_list)

    compiled_arr = np.stack([predictions_list, labels_list, race_list]).T
    compiled_df = pd.DataFrame(data=compiled_arr, columns=['emote_predictions', 'emote_groundtruth', 'race_predictions'])
    compiled_df.to_csv('MicroEthnic-A1.csv', index=False)

    f1 = f1_score(y_true=labels_list, y_pred=predictions_list, zero_division=0.0, average='macro')
    f1_per_emote = f1_score(y_true=labels_list, y_pred=predictions_list, average=None, zero_division=0.0)
    
    table_dat = [f1_per_emote[0], f1_per_emote[1], f1_per_emote[2], f1]
    # table_dat = [f1_per_emote[0], f1_per_emote[1], f1_per_emote[2], f1_per_emote[3], f1_per_emote[4], f1]
    # table_dat = [f1_per_emote[0], f1_per_emote[1], f1_per_emote[2], f1_per_emote[3], f1_per_emote[4], f1_per_emote[5], f1_per_emote[6], \
    #             f1_per_emote[7], f1]    
    table.add_row(table_dat)   
    print("Predictions")
    print(predictions_list)
    print("Groundtruth")
    print(labels_list)



    print(table)


if __name__ == "__main__":
    # weights_name = '/scratch/project_2005312/ice/Weights/BP4D/bp4d_videomae_cropped_notaligned' # MAHTI
    # weights_name = '/scratch/project_462000442/ice/Weights/BP4D/bp4d_videomae_aligned' # LUMI

    # # weights path (lumi-cloud)
    # weights_name = '/scratch/project_462000772/ice/Weights/BP4D/A-BP4D-1'

    # # weights path (local)
    # weights_name = '/home/hq/Documents/Weights/Upstream(BP4D)/random'


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights_path', type=str, help='The name of the weights', default='/home/hq/Documents/Weights/MicroEthnic')
    parser.add_argument('--weights_name', type=str, help='The name of the weights', default='MicroEthnic-A1')
    parser.add_argument('--root_dir', type=str, help='The name of the h5py', default="/home/hq/Documents/data/SAMM/SAMM_CROP")
    parser.add_argument('--annotation_file', type=str, help='The directory that stores the labels', default="/home/hq/Documents/WorkingRepos/AU_Localization/CD6ME_Ethnic/JointDB_MetaEmotionConcised.csv")
    parser.add_argument('--epochs', type=int, help='Epoch to run', default=15)
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=32)     
    # parser.add_argument('--fold_selection', type=int, help='select the fold to train/test (0/1/2)', default=0)      

    # Parse the arguments
    args = parser.parse_args()    

    if os.path.exists(args.weights_name) == False:
        os.makedirs(args.weights_name)

    start_time = time.time()
    test_samm(args=args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    print(f'Time: {elapsed_time:.2f} seconds')    