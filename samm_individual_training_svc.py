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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from samm_dataset import *
from samm_models import TinyVIT, res18_imagenet, CultureAwareClassifier, CulturalMTL, CulturalDualNetwork_LateFusion, CulturalDualNetwork_TinyViT_LateFusion, CulturalDualNetwork_ResViT_LateFusion
from samm_models import res18_imagenet_features

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



def train_samm(args):
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

    # mlflow setup
    mlflow_experiment_name = "{}/{}".format(weights_name, "mlflow")
    if os.path.exists(mlflow_experiment_name) == False:
        os.makedirs(mlflow_experiment_name)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run(run_name=mlflow_experiment_name)

    weights_path = os.path.join(weights_path, weights_name)


    # Get all LOSO splits
    sampled_asian_df, sampled_non_asian_df, sampled_mixed_df = racial_based_sampling(annotation_file, concised_race=False)
    # splits = get_loso_splits(sampled_asian_df)
    splits = get_loso_splits(sampled_non_asian_df)
    # splits = get_loso_splits(sampled_mixed_df)
    # splits = get_loso_splits(annotation_file)

    racially_sampled_df = sampled_non_asian_df


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
    # table.field_names = ['Subject', 'loss', 'Positive', 'Negative', 'Surprise', 'avg_f1']
    table.field_names = ['Subject', 'loss', 'Non-negative', 'Negative', 'avg_f1']
    softmax = nn.Softmax(dim=1)

    all_folds_predictions_list = []
    all_folds_labels_list = []   

    all_folds_ace = []
    all_folds_mape = []

    for subject_out, (train_data, test_data) in splits.items():
        clf = SVC(C=100)
        # clf = RandomForestClassifier(n_estimators=100, random_state=42)

        print(f"Leaving out subject: {subject_out}")
        
        # Filtering based on LOSO protocol
        train_data = racially_sampled_df[racially_sampled_df['Subject'] != subject_out]  # Train set
        test_data = racially_sampled_df[racially_sampled_df['Subject'] == subject_out]  # Test set
        # self.index = self.data.index.values
        # self.test_index = self.test_data.index.values        

        # Train and Test datasets
        train_dataset = JointDBDataset_RacialDriven(root_dir, train_data, train_test_flag='train', subject_out=subject_out, transform=transform, of_transform=optical_flow_transform)
        test_dataset = JointDBDataset_RacialDriven(root_dir, test_data, train_test_flag='test', subject_out=subject_out, transform=transform, of_transform=optical_flow_transform)

        # train_dataset = SAMMDataset(root_dir, annotation_file, train_test_flag='train', subject_out=subject_out, transform=transform, of_transform=optical_flow_transform)
        # test_dataset = SAMMDataset(root_dir, annotation_file, train_test_flag='test', subject_out=subject_out, transform=transform, of_transform=optical_flow_transform)

        # Data Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # model
        # model = TinyVIT(num_classes=3).cuda()
        model = res18_imagenet_features(num_classes=2).cuda()
        model = model.eval()

        with torch.no_grad():
            predictions_list = []
            labels_list = []   
            one_hot_predictions_list = []
            racial_list = []

            # Iterate through training data
            for batch in train_loader:
                frames, of_frames, labels, race = batch
                frames = frames.cuda()
                of_frames = of_frames.float().cuda() 
                # of_frames = of_frames.permute(0, 3, 1, 2)
                labels = labels.cuda()
                labels = labels.to(torch.int64)
                race = race.to(torch.int64).cuda()

                predicted_emotions = model(of_frames)
                # of_frames = of_frames.view(of_frames.size(0), -1).cpu().numpy()
                clf.fit(predicted_emotions.cpu().numpy(), labels.cpu().numpy())

   
                # predicted_emotions = softmax(predicted_emotions)
                # one_hot_p_emotions = torch.nn.functional.one_hot(torch.argmax(predicted_emotions, dim=1), num_classes=3).cpu().detach().numpy()
                # predicted_emotions = torch.argmax(predicted_emotions, 1, keepdim=True) 


                # one_hot_predictions_list += [one_hot_p_emotions]
            #     predictions_list += [predicted_emotions.cpu().numpy()]
            #     labels_list += [labels.cpu().numpy()]     
            #     # racial_list += [race_logits.cpu().numpy()]


            # predictions_list = np.concatenate(predictions_list)
            # labels_list = np.concatenate(labels_list)
            # # racial_list = np.concatenate(racial_list)
            # one_hot_predictions_list = np.concatenate(one_hot_predictions_list)
            # acc = f1_score(y_true=labels_list, y_pred=predictions_list, zero_division=0.0, average='micro')
            # f1 = f1_score(y_true=labels_list, y_pred=predictions_list, zero_division=0.0, average='macro')
            # f1_per_emote = f1_score(y_true=labels_list, y_pred=predictions_list, average=None, zero_division=0.0)






        with torch.no_grad():

            for batch in test_loader:
                frames, of_frames, labels, race = batch
                frames = frames.cuda()
                of_frames = of_frames.float().cuda()

                labels = labels.cuda()      
                labels = labels.to(torch.int64)
                race = race.to(torch.int64).cuda()


                predicted_emotions = model(of_frames)
                # of_frames = of_frames.view(of_frames.size(0), -1).cpu().numpy()
                predicted_emotions = clf.predict(predicted_emotions.cpu().numpy())

                # predicted_emotions = torch.argmax(predicted_emotions, 1) 
                all_folds_predictions_list += [predicted_emotions]
                all_folds_labels_list += [labels.cpu().numpy()]     

                # all_folds_predictions_list += [predicted_emotions.cpu().numpy()]


     

    table = PrettyTable()
    # table.field_names = ['Fear', 'Other', 'Surprise', 'Happiness', 'Anger', 'Disgust', 'Contempt', 'Sadness', 'avg_f1']
    # table.field_names = ['Other', 'Surprise', 'Happiness', 'Anger', 'Contempt', 'avg_f1']
    # table.field_names = ['Positive', 'Negative', 'Surprise', 'avg_f1']
    table.field_names = ['Non-Negative', 'Negative', 'avg_f1']
    predictions_list = np.concatenate(all_folds_predictions_list)
    labels_list = np.concatenate(all_folds_labels_list)
    f1 = f1_score(y_true=labels_list, y_pred=predictions_list, zero_division=0.0, average='macro')
    f1_per_emote = f1_score(y_true=labels_list, y_pred=predictions_list, average=None, zero_division=0.0)
    
    # table_dat = [f1_per_emote[0], f1_per_emote[1], f1_per_emote[2], f1]
    table_dat = [f1_per_emote[0], f1_per_emote[1], f1]
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
    parser.add_argument('--weights_name', type=str, help='The name of the weights', default='MicroEthnic-A1-mixedEthnic')
    parser.add_argument('--root_dir', type=str, help='The name of the h5py', default="/home/hq/Documents/data/SAMM/SAMM_CROP")
    parser.add_argument('--annotation_file', type=str, help='The directory that stores the labels', default="/home/hq/Documents/WorkingRepos/AU_Localization/CD6ME_Ethnic/JointDB_MetaEmotionConcised.csv")
    parser.add_argument('--epochs', type=int, help='Epoch to run', default=15)
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=128)     
    # parser.add_argument('--fold_selection', type=int, help='select the fold to train/test (0/1/2)', default=0)      

    # Parse the arguments
    args = parser.parse_args()    

    if os.path.exists(args.weights_name) == False:
        os.makedirs(args.weights_name)

    start_time = time.time()
    train_samm(args=args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    print(f'Time: {elapsed_time:.2f} seconds')     