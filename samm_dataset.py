import torch
import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import LeaveOneGroupOut
from natsort import natsorted

class JointDBDataset_RacialDriven(Dataset):
    def __init__(self, root_dir, annotation_file, train_test_flag, subject_out=None, transform=None, of_transform=None):
        """
        Args:
            root_dir (str): Path to JointDB dataset folder.
            annotation_file (str): Path to the annotations CSV file.
            subject_out (str): Subject ID to leave out (for testing in LOSO).
            transform (torchvision.transforms): Transformations for data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.of_transform = of_transform

        # Load metadata
        self.annotations = annotation_file

        # Map emotion labels to integers
        emotion_labels = ['Positive', 'Negative', 'Surprise']    
        self.emotion_to_int = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

        race_labels = ['Asian', 'Non-Asian']
        self.race_to_int = {race: idx for idx, race in enumerate(race_labels)}

        self.annotations['Estimated Emotion Quantized'] = self.annotations['EstimatedEmotionConcised'].map(self.emotion_to_int)
        self.annotations['Estimated Emotion Quantized'] = self.annotations['Estimated Emotion Quantized'].replace({2: 0}) # to simplify race into non-negative and negative
        self.drop_index = self.annotations.loc[self.annotations['Estimated Emotion Quantized'].isna()].index.values
        self.annotations = self.annotations.loc[self.annotations['Estimated Emotion Quantized'].notna()] # drop rows, remember to drop similar things in optical flow
        self.annotations = self.annotations.reset_index()

        self.annotations['RaceConcised'] = self.annotations['RaceConcised'].map(self.race_to_int)


        # Load optical flow if available
        npy_casme2_of = "/home/hq/Documents/data/CASME2/casme2_uv_frames_secrets_of_OF.npy"
        npy_samm_of = "/home/hq/Documents/data/SAMM/samm_uv_frames_secrets_of_OF.npy"

        resized_flow = []
        self.optical_flow_casme2 = np.load(npy_casme2_of)
        self.optical_flow_samm = np.load(npy_samm_of)

        for img in self.optical_flow_casme2:
            img = np.transpose(img, (1, 2, 0))  # to H, W, C
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1))  # to C, H, W
            resized_flow += [img]
        for img in self.optical_flow_samm:
            img = np.transpose(img, (1, 2, 0))  # to H, W, C
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1))  # to C, H, W
            resized_flow += [img]

        self.optical_flow = np.stack(resized_flow)
        self.optical_flow = np.delete(self.optical_flow, self.drop_index, axis=0)

        # sampled_asian_df, sampled_non_asian_df, sampled_mixed_df = racial_based_sampling(self.annotations, concised_race=True)
        # self.annotations = sampled_asian_df
        # self.annotations = sampled_non_asian_df
        # self.annotations = sampled_mixed_df

        # # Filtering based on LOSO protocol
        # self.data = self.annotations[self.annotations['Subject'] != subject_out]  # Train set
        # self.test_data = self.annotations[self.annotations['Subject'] == subject_out]  # Test set
        # self.index = self.data.index.values
        # self.test_index = self.test_data.index.values
        self.index = self.annotations.index.values
        self.test_index = self.annotations.index.values
        self.test_data = self.annotations
        self.data = self.annotations

        if train_test_flag == 'test':
            self.data = self.test_data
            self.index = self.test_index

        if self.optical_flow is not None:
            self.of_frame = self.optical_flow[self.index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject = row['Subject']
        sequence = row['Filename']
        label = row['Estimated Emotion Quantized']  # Expression label
        race = row['RaceConcised']
        gender = row['Gender']

        # Construct image path
        if row['Dataset'] == 'CASME2':
            data_path = '/home/hq/Documents/data/CASME2/Cropped'
            onset_frame = 'OnsetFrame'
            apex_frame = 'ApexFrame'
            offset_frame = 'OffsetFrame'
        else:
            data_path = '/home/hq/Documents/data/SAMM/SAMM_CROP'
            onset_frame = 'Onset'
            apex_frame = 'Apex Frame'
            offset_frame = 'Offset'            
        

        # img_folder = os.path.join(self.root_dir, row['Dataset'], subject, sequence)
        img_folder = os.path.join(data_path, subject, sequence)
        frame_files = natsorted(os.listdir(img_folder))  # Sort frames  

        if row['Dataset'] == 'CASME2':           
            onset_frame = [x for x in frame_files if 'reg_img' + str(int(row['OnsetFrame'])) in x]
            offset_frame = [x for x in frame_files if 'reg_img' + str(int(row['OffsetFrame'])) in x]
            if row['ApexFrame'] == '/':
                apex_frame = ['reg_img' + str(int((int(row['OnsetFrame']) + int(row['OffsetFrame'])) / 2)) + '.jpg']
            else:
                apex_frame = [x for x in frame_files if 'reg_img' + str(int(row['ApexFrame'])) in x]
        else:
            onset_frame = [x for x in frame_files if str(int(row['Onset'])) in x]
            apex_frame = [x for x in frame_files if str(int(row['Apex Frame'])) in x]
            offset_frame = [x for x in frame_files if str(int(row['Offset'])) in x]

        frames = []

        if len(apex_frame) == 0:
            # take offset as apex if apex label is -1
            apex_frame = offset_frame
        onset_frame = onset_frame[0]
        apex_frame = apex_frame[0]
        offset_frame = offset_frame[0]

        # frame_files = [onset_frame, apex_frame]
        # for file in frame_files:
        #     img_path = os.path.join(img_folder, file)
        #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #     if self.transform:
        #         img = self.transform(img)
            
        #     frames.append(img)
        # frames = torch.stack(frames)  

        img_path = os.path.join(img_folder, apex_frame)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            frames = self.transform(img)        

        # Convert list of tensors to a single tensor

        # read optical flow if available
        if self.optical_flow is not None:
            of_frame = self.of_frame[idx]
            of_frame = np.transpose(of_frame, (1, 2, 0))
            if self.of_transform:
                of_frame = self.of_transform(of_frame)
        else:
            of_frame = None
        
        return frames, of_frame, label, race  # Returning sequence of frames and label


class JointDBDataset(Dataset):
    def __init__(self, root_dir, annotation_file, train_test_flag, subject_out=None, transform=None, of_transform=None):
        """
        Args:
            root_dir (str): Path to JointDB dataset folder.
            annotation_file (str): Path to the annotations CSV file.
            subject_out (str): Subject ID to leave out (for testing in LOSO).
            transform (torchvision.transforms): Transformations for data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.of_transform = of_transform

        # Load metadata
        self.annotations = annotation_file

        # Map emotion labels to integers
        emotion_labels = ['Positive', 'Negative', 'Surprise']
        self.emotion_to_int = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

        race_labels = ['Asian', 'Non-Asian']
        self.race_to_int = {race: idx for idx, race in enumerate(race_labels)}

        self.annotations['Estimated Emotion Quantized'] = self.annotations['EstimatedEmotionConcised'].map(self.emotion_to_int)
        self.drop_index = self.annotations.loc[self.annotations['Estimated Emotion Quantized'].isna()].index.values
        self.annotations = self.annotations.loc[self.annotations['Estimated Emotion Quantized'].notna()] # drop rows, remember to drop similar things in optical flow
        self.annotations = self.annotations.reset_index()

        self.annotations['RaceConcised'] = self.annotations['RaceConcised'].map(self.race_to_int)


        # Load optical flow if available
        npy_casme2_of = "/home/hq/Documents/data/CASME2/casme2_uv_frames_secrets_of_OF.npy"
        npy_samm_of = "/home/hq/Documents/data/SAMM/samm_uv_frames_secrets_of_OF.npy"

        resized_flow = []
        self.optical_flow_casme2 = np.load(npy_casme2_of)
        self.optical_flow_samm = np.load(npy_samm_of)

        for img in self.optical_flow_casme2:
            img = np.transpose(img, (1, 2, 0))  # to H, W, C
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1))  # to C, H, W
            resized_flow += [img]
        for img in self.optical_flow_samm:
            img = np.transpose(img, (1, 2, 0))  # to H, W, C
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1))  # to C, H, W
            resized_flow += [img]

        self.optical_flow = np.stack(resized_flow)
        self.optical_flow = np.delete(self.optical_flow, self.drop_index, axis=0)

        # sampled_asian_df, sampled_non_asian_df, sampled_mixed_df = racial_based_sampling(self.annotations, concised_race=True)
        # self.annotations = sampled_asian_df
        # self.annotations = sampled_non_asian_df
        # self.annotations = sampled_mixed_df

        # Filtering based on LOSO protocol
        self.data = self.annotations[self.annotations['Subject'] != subject_out]  # Train set
        self.test_data = self.annotations[self.annotations['Subject'] == subject_out]  # Test set
        self.index = self.data.index.values
        self.test_index = self.test_data.index.values

        if train_test_flag == 'test':
            self.data = self.test_data
            self.index = self.test_index

        if self.optical_flow is not None:
            self.of_frame = self.optical_flow[self.index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject = row['Subject']
        sequence = row['Filename']
        label = row['Estimated Emotion Quantized']  # Expression label
        race = row['RaceConcised']
        gender = row['Gender']

        # Construct image path
        if row['Dataset'] == 'CASME2':
            data_path = '/home/hq/Documents/data/CASME2/Cropped'
            onset_frame = 'OnsetFrame'
            apex_frame = 'ApexFrame'
            offset_frame = 'OffsetFrame'
        else:
            data_path = '/home/hq/Documents/data/SAMM/SAMM_CROP'
            onset_frame = 'Onset'
            apex_frame = 'Apex Frame'
            offset_frame = 'Offset'            
        

        # img_folder = os.path.join(self.root_dir, row['Dataset'], subject, sequence)
        img_folder = os.path.join(data_path, subject, sequence)
        frame_files = natsorted(os.listdir(img_folder))  # Sort frames  

        if row['Dataset'] == 'CASME2':           
            onset_frame = [x for x in frame_files if 'reg_img' + str(int(row['OnsetFrame'])) in x]
            offset_frame = [x for x in frame_files if 'reg_img' + str(int(row['OffsetFrame'])) in x]
            if row['ApexFrame'] == '/':
                apex_frame = ['reg_img' + str(int((int(row['OnsetFrame']) + int(row['OffsetFrame'])) / 2)) + '.jpg']
            else:
                apex_frame = [x for x in frame_files if 'reg_img' + str(int(row['ApexFrame'])) in x]
        else:
            onset_frame = [x for x in frame_files if str(int(row['Onset'])) in x]
            apex_frame = [x for x in frame_files if str(int(row['Apex Frame'])) in x]
            offset_frame = [x for x in frame_files if str(int(row['Offset'])) in x]

        frames = []

        if len(apex_frame) == 0:
            # take offset as apex if apex label is -1
            apex_frame = offset_frame
        onset_frame = onset_frame[0]
        apex_frame = apex_frame[0]
        offset_frame = offset_frame[0]

        # frame_files = [onset_frame, apex_frame]
        # for file in frame_files:
        #     img_path = os.path.join(img_folder, file)
        #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #     if self.transform:
        #         img = self.transform(img)
            
        #     frames.append(img)
        # frames = torch.stack(frames)  

        img_path = os.path.join(img_folder, apex_frame)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            frames = self.transform(img)        

        # Convert list of tensors to a single tensor

        # read optical flow if available
        if self.optical_flow is not None:
            of_frame = self.of_frame[idx]
            of_frame = np.transpose(of_frame, (1, 2, 0))
            if self.of_transform:
                of_frame = self.of_transform(of_frame)
        else:
            of_frame = None
        
        return frames, of_frame, label, race  # Returning sequence of frames and label


class SAMMDataset(Dataset):
    def __init__(self, root_dir, annotation_file, train_test_flag, subject_out=None, transform=None, of_transform=None):
        """
        Args:
            root_dir (str): Path to SAMM dataset folder.
            annotation_file (str): Path to the annotations CSV file.
            subject_out (str): Subject ID to leave out (for testing in LOSO).
            transform (torchvision.transforms): Transformations for data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.of_transform = of_transform

        # Load metadata
        self.annotations = pd.read_excel(annotation_file)
        # self.annotations = self.annotations.loc[~self.annotations['Estimated Emotion'].isin(['Fear', 'Disgust', 'Sadness'])]

        # Map emotion labels to integers
        # emotion_labels = ['Fear', 'Other', 'Surprise', 'Happiness', 'Anger', 'Disgust', 'Contempt', 'Sadness']
        # emotion_labels = ['Other', 'Surprise', 'Happiness', 'Anger', 'Contempt']
        emotion_labels = ['Positive', 'Negative', 'Surprise']
        self.emotion_to_int = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

        race_labels = ['Asian', 'Non-Asian']
        self.race_to_int = {race: idx for idx, race in enumerate(race_labels)}

        # self.annotations['Estimated Emotion Quantized'] = self.annotations['Estimated Emotion'].map(self.emotion_to_int)
        self.annotations['Estimated Emotion Quantized'] = self.annotations['EstimatedEmotionConcised'].map(self.emotion_to_int)
        self.annotations = self.annotations.loc[self.annotations['Estimated Emotion Quantized'].notna()]

        self.annotations['RaceConcised'] = self.annotations['RaceConcised'].map(self.race_to_int)

        # load optical flow
        npy_filename = "/home/hq/Documents/data/SAMM/samm_uv_frames_secrets_of_OF.npy"
        self.optical_flow = np.load(npy_filename)   
        resized_flow = []
        for img in self.optical_flow:
            img = np.transpose(img, (1, 2, 0)) # to H, W, C
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1)) # to C, H, W
            resized_flow += [img]
        self.optical_flow = np.stack(resized_flow)
        

        # Filtering based on LOSO protocol
        self.data = self.annotations[self.annotations['Subject'] != subject_out]  # Train set
        self.test_data = self.annotations[self.annotations['Subject'] == subject_out]  # Test set
        self.index = self.data.index.values
        self.test_index = self.test_data.index.values

        if train_test_flag == 'test':
            self.data = self.test_data
            self.index = self.test_index

        self.of_frame = self.optical_flow[self.index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject = row['Subject'].astype(str).zfill(3)
        sequence = row['Filename']
        label = row['Estimated Emotion Quantized']  # Expression label
        race = row['RaceConcised']

        # Construct image path
        img_folder = os.path.join(self.root_dir, subject, sequence)
        frame_files = natsorted(os.listdir(img_folder))  # Sort frames
        frames = []

        onset_frame = [x for x in frame_files if row['Onset'].astype(str) in x]
        apex_frame = [x for x in frame_files if row['Apex Frame'].astype(str) in x]
        offset_frame = [x for x in frame_files if row['Offset'].astype(str) in x]

        if len(apex_frame) == 0:
            # take offset as apex if apex label is -1
            apex_frame = offset_frame
        onset_frame = onset_frame[0]
        apex_frame = apex_frame[0]
        offset_frame = offset_frame[0]

        frame_files = [onset_frame, apex_frame]

        for file in frame_files:
            img_path = os.path.join(img_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)
            
            frames.append(img)

        frames = torch.stack(frames)  # Convert list of tensors to a single tensor

        # read optical flow
        of_frame = self.of_frame[idx]
        of_frame = np.transpose(of_frame, (1, 2, 0))
        if self.of_transform:
            of_frame = self.of_transform(of_frame)
        
        return frames, of_frame, label, race  # Returning sequence of frames and label

def get_loso_splits(annotation_file):
    """
    Creates Leave-One-Subject-Out (LOSO) splits.
    """
    if type(annotation_file) == str:
        df = pd.read_excel(annotation_file)
    else:
        df = annotation_file
    logo = LeaveOneGroupOut()

    # emotion_classes_omittance = ['Other']
    # df = df.loc[~df['Estimated Emotion'].isin(emotion_classes_omittance)]
    
    subjects = df['Subject'].unique()
    subjects = sorted(subjects)
    splits = {}

    for train_idx, test_idx in logo.split(df, groups=df['Subject']):
        test_subject = df.iloc[test_idx]['Subject'].unique()[0]
        splits[test_subject] = (df.iloc[train_idx], df.iloc[test_idx])

    return splits




def emote_downsampling(df):
    # Down-sample Negative in sampled_asian_df
    positive_df = df[df['EstimatedEmotionConcised'] == 'Positive']
    surprise_df = df[df['EstimatedEmotionConcised'] == 'Surprise']
    negative_df = df[df['EstimatedEmotionConcised'] == 'Negative']
    # Calculate target size for Negative
    target_negative_size = 192
    # Sample Negative to match the target size
    sampled_negative_df = negative_df.sample(n=target_negative_size, random_state=42)
    # Combine Positive, Surprise, and sampled Negative
    df = pd.concat([positive_df, surprise_df, sampled_negative_df], ignore_index=True)
    return df



def racial_based_sampling(annotations, concised_race=False):
    # if concised_race:
    #     race_labels = ['Asian', 'Non-Asian']
    #     race_to_int = {race: idx for idx, race in enumerate(race_labels)}
    #     annotations['RaceConcised'] = annotations['RaceConcised'].map(race_to_int)

    annotations = annotations.loc[annotations['EstimatedEmotionConcised'].isin(['Positive', 'Negative', 'Surprise'])]
    # annotations = emote_downsampling(df=annotations)

    # racial-based sampling
    if concised_race == False:
        asian_df = annotations.loc[annotations['RaceConcised']=='Asian']
        non_asian_df = annotations.loc[annotations['RaceConcised']=='Non-Asian']
    # racial-based sampling
    else:
        asian_df = annotations.loc[annotations['RaceConcised']==0]
        non_asian_df = annotations.loc[annotations['RaceConcised']==1]

    # asian df sampling
    np.random.seed(42) 
    # asian_df = asian_df[asian_df['Subject'].str.startswith('sub')]
    asian_subjects = asian_df['Subject'].unique() 
    sampled_asian_subjects = np.random.choice(asian_subjects, size=16, replace=False)  # Randomly sample 16 subjects
    sampled_asian_df = asian_df[asian_df['Subject'].isin(sampled_asian_subjects)]
    # sampled_asian_df = sampled_asian_df.reset_index(drop=True)

    # non-asian df sampling
    non_asian_subjects = non_asian_df['Subject'].unique() 
    sampled_non_asian_subjects = np.random.choice(non_asian_subjects, size=16, replace=False)  # Randomly sample 16 subjects
    sampled_non_asian_df = non_asian_df[non_asian_df['Subject'].isin(sampled_non_asian_subjects)]
    # sampled_non_asian_df = sampled_non_asian_df.reset_index(drop=True)

    # sample 8 subjects from non-asian df, 8 subjects from asian_df
    sampled_non_asian_subjects = np.random.choice(non_asian_subjects, size=8, replace=False)  # Randomly sample 16 subjects
    sampled_asian_subjects = np.random.choice(asian_subjects, size=8, replace=False)
    sampled_mixed_non_asian_df = non_asian_df[non_asian_df['Subject'].isin(sampled_non_asian_subjects)]
    sampled_mixed_asian_df = asian_df[asian_df['Subject'].isin(sampled_asian_subjects)]
    sampled_mixed_df = pd.concat([sampled_mixed_non_asian_df, sampled_mixed_asian_df], ignore_index=True)
    # sampled_mixed_df = sampled_mixed_df.reset_index(drop=True)   

    # # Calculate target size for Negative
    # sampled_asian_df = emote_downsampling(df=sampled_asian_df)
    # sampled_non_asian_df = emote_downsampling(df=sampled_non_asian_df)
    # sampled_mixed_df = emote_downsampling(df=sampled_mixed_df)


    print(sampled_asian_df['EstimatedEmotionConcised'].value_counts())
    print(sampled_non_asian_df['EstimatedEmotionConcised'].value_counts())
    print(sampled_mixed_df['EstimatedEmotionConcised'].value_counts())
    a = 1

    return sampled_asian_df, sampled_non_asian_df, sampled_mixed_df