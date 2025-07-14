import torch
import torch.nn as nn

from marlin_pytorch import Marlin
from marlin_pytorch.config import register_model_from_yaml

from models_utilities import count_parameters
from tiny_vit import tiny_vit_21m_224
from tiny_vit import tiny_vit_5m_224, tiny_vit_21m_384, tiny_vit_21m_512
from torchvision.models import resnet18, ResNet18_Weights

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

from torchvision.models import swin_b



class TinyVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # for eventual downstream classification
        linear_layer = nn.Linear(in_features=576, out_features=num_classes, bias=True)
        # linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)

        self.flatten = nn.Flatten()
        self.fc = linear_layer   

        self.model = tiny_vit_21m_224(pretrained=True)
        # self.model = tiny_vit_5m_224(pretrained=True)      
        # self.model = tiny_vit_21m_384(pretrained=True)        
        # self.model = tiny_vit_21m_512(pretrained=True)

        # for param in self.model.parameters():
        #     param.requires_grad = False   

        self.model.head = linear_layer

        # self.softmax = nn.Softmax(dim=1)
        count_parameters(self.model)
        
    def forward(self, x):
        output = self.model(x)
        # output = self.softmax(output)
        return output
    

class res18_imagenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.spatial_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        linear_layer = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # self.spatial_model.fc = linear_layer

        # self.spatial_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # linear_layer = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

        # # weights freezing
        # for param in self.spatial_model.parameters():
        #     param.requires_grad = False      
        # for param in self.spatial_model.layer4.parameters():
        #     param.requires_grad = True    

        self.spatial_model.fc = linear_layer        
        count_parameters(self.spatial_model)

    def forward(self, x):
        x = self.spatial_model(x)
        return x

class res18_imagenet_features(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.spatial_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model = nn.Sequential(*list(self.spatial_model.children())[:-1]).cuda()
        
        self.flatten = nn.Flatten()     
        count_parameters(self.spatial_model)

    def forward(self, x):
        x = self.spatial_model(x)
        x = self.flatten(x)
        return x    

    
class CultureAwareClassifier(nn.Module):
    def __init__(self, num_classes, num_cultures):
        super().__init__()
        self.spatial_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        linear_layer = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # culture-specific bias embeddings
        self.culture_bias = nn.Embedding(num_cultures, num_classes)
        # Optionally, initialize culture bias near zero
        nn.init.zeros_(self.culture_bias.weight)

        self.spatial_model.fc = linear_layer        
        count_parameters(self.spatial_model)
    
    def forward(self, image, culture_id):
        image_feat = self.spatial_model(image)
        # logits = self.classifier(image_feat)
        # Add the culture-specific bias for the given culture
        culture_offset = self.culture_bias(culture_id)  # shape: [batch, num_classes]
        logits = image_feat + culture_offset
        probs = torch.softmax(logits, dim=-1)        
        return probs

class CulturalMTL(nn.Module):
    def __init__(self, num_classes, num_cultures):
        super().__init__()
        self.spatial_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()

        # Remove the last fully connected layer
        self.spatial_model = nn.Sequential(*list(self.spatial_model.children())[:-1])
        self.flatten = nn.Flatten()
        

        self.linear_layer_emotion = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # self.spatial_model_emotion.fc = linear_layer_emotion

        # self.spatial_model_race = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.linear_layer_race = nn.Linear(in_features=512, out_features=num_cultures, bias=True)
        # self.spatial_model_race.fc = linear_layer_race

        count_parameters(self.spatial_model)
        # count_parameters(self.spatial_model_race)
    
    def forward(self, image):
        image_feat = self.spatial_model(image)
        image_feat = self.flatten(image_feat)
        emote_logit = self.linear_layer_emotion(image_feat)
        race_logit = self.linear_layer_race(image_feat)
        # logits = self.classifier(image_feat)
        # Add the culture-specific bias for the given culture
        # culture_offset = self.culture_bias(culture_id)  # shape: [batch, num_classes]
        # logits = emote_logit + race_logit
        # probs = torch.softmax(logits, dim=-1)        
        return emote_logit, race_logit


class CulturalDualNetwork_LateFusion(nn.Module):
    def __init__(self, num_classes, num_cultures, num_gender=2):
        super().__init__()
        self.spatial_model_emote = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model_race = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model_gender = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        

        # Remove the last fully connected layer from both models
        self.spatial_model_emote = nn.Sequential(*list(self.spatial_model_emote.children())[:-1])
        self.spatial_model_race = nn.Sequential(*list(self.spatial_model_race.children())[:-1])
        self.spatial_model_gender = nn.Sequential(*list(self.spatial_model_gender.children())[:-1])
        
        self.flatten = nn.Flatten()
        
        # Linear layers for emotion and race
        self.linear_layer_emotion = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.linear_layer_race = nn.Linear(in_features=512, out_features=num_cultures, bias=True)
        self.linear_layer_gender = nn.Linear(in_features=512, out_features=num_gender, bias=True)
        

        # Final linear layer after merging features
        self.final_linear_layer = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        # self.final_linear_layer = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
                   
        
        count_parameters(self.spatial_model_emote)
        count_parameters(self.spatial_model_race)
        count_parameters(self.spatial_model_gender)
    
    def forward(self, image, image_rgb):
        # Extract features from both models
        feat_emote = self.spatial_model_emote(image)
        feat_race = self.spatial_model_emote(image)
        feat_gender = self.spatial_model_gender(image)
        
        
        # Flatten the features
        feat_emote = self.flatten(feat_emote)
        feat_race = self.flatten(feat_race)
        # feat_gender = self.flatten(feat_gender)
        
        # Compute logits for emotion and race
        emote_logit = self.linear_layer_emotion(feat_emote)
        race_logit = self.linear_layer_race(feat_race) 
        # gender_logit = self.linear_layer_gender(feat_gender)

        # Merge the features
        merged_feat = torch.cat((feat_emote, feat_race), dim=1)
        # merged_feat = torch.cat((feat_emote, feat_race, feat_gender), dim=1)
        
        # Compute final logits
        final_logit = self.final_linear_layer(merged_feat)
        
        # Compute probabilities
        # emote_probs = torch.softmax(emote_logit, dim=-1)
        # race_probs = torch.softmax(race_logit, dim=-1)
        # final_probs = torch.softmax(final_logit, dim=-1)
                

        return emote_logit, race_logit, final_logit


class CulturalDualNetwork_TinyViT_LateFusion(nn.Module):
    def __init__(self, num_classes, num_cultures, num_gender=2):
        super().__init__()


        self.spatial_model_emote = tiny_vit_21m_224(pretrained=True).cuda()
        self.spatial_model_race = tiny_vit_21m_224(pretrained=True).cuda()
        
        # self.spatial_model_emote = tiny_vit_5m_224(pretrained=True)  
        # self.spatial_model_race = tiny_vit_5m_224(pretrained=True)  

        self.emote_linear_layer = nn.Linear(in_features=576, out_features=num_classes, bias=True)
        self.race_linear_layer = nn.Linear(in_features=576, out_features=num_cultures, bias=True)
        # self.emote_linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)
        # self.race_linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)        

        # self.spatial_model_emote.head = emote_linear_layer
        # self.spatial_model_race.head = race_linear_layer
        self.final_linear_layer = nn.Linear(in_features=576, out_features=num_classes, bias=True)
        # self.final_linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)
        
        count_parameters(self.spatial_model_emote)
        count_parameters(self.spatial_model_race)

    
    def forward(self, image):
        # Extract features from both models
        feat_emote = self.spatial_model_emote.forward_features(image)
        feat_emote = self.spatial_model_emote.norm_head(feat_emote)

        feat_race = self.spatial_model_race.forward_features(image)
        feat_race = self.spatial_model_race.norm_head(feat_race)

        emote_logit = self.emote_linear_layer(feat_emote)
        race_logit = self.race_linear_layer(feat_race)
        

        # Merge the features
        merged_feat = torch.mean(torch.stack([feat_emote, feat_race]), dim=0)
        # merged_feat = torch.cat((feat_emote, feat_race), dim=1)
        
        # Compute final logits
        final_logit = self.final_linear_layer(merged_feat)
        
        # Compute probabilities
        # emote_probs = torch.softmax(emote_logit, dim=-1)
        # race_probs = torch.softmax(race_logit, dim=-1)
        # final_probs = torch.softmax(final_logit, dim=-1)
                

        return emote_logit, race_logit, final_logit

class CulturalDualNetwork_ResViT_LateFusion(nn.Module):
    def __init__(self, num_classes, num_cultures):
        super().__init__()


        # self.spatial_model_emote = tiny_vit_21m_224(pretrained=True).cuda()
        self.spatial_model_emote = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model_race = tiny_vit_21m_224(pretrained=True).cuda()
        
        # self.spatial_model_emote = tiny_vit_5m_224(pretrained=True).cuda()  
        # self.spatial_model_race = tiny_vit_5m_224(pretrained=True).cuda()  

        self.spatial_model_emote = nn.Sequential(*list(self.spatial_model_emote.children())[:-1])
        self.flatten = nn.Flatten()
        self.linear_layer_emotion = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        # self.emote_linear_layer = nn.Linear(in_features=576, out_features=num_classes, bias=True)
        self.race_linear_layer = nn.Linear(in_features=576, out_features=num_cultures, bias=True)
        # self.emote_linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)
        # self.race_linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)        

        # self.spatial_model_emote.head = emote_linear_layer
        # self.spatial_model_race.head = race_linear_layer
        self.final_linear_layer = nn.Linear(in_features=1088, out_features=num_classes, bias=True)
        # self.final_linear_layer = nn.Linear(in_features=832, out_features=num_classes, bias=True)
        # self.final_linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)
        
        count_parameters(self.spatial_model_emote)
        count_parameters(self.spatial_model_race)

    
    def forward(self, image, image_rgb):
        # Extract features from both models
        feat_emote = self.spatial_model_emote(image)
        feat_emote = self.flatten(feat_emote)

        feat_race = self.spatial_model_race.forward_features(image_rgb)
        feat_race = self.spatial_model_race.norm_head(feat_race)

        emote_logit = self.linear_layer_emotion(feat_emote)
        race_logit = self.race_linear_layer(feat_race)
        

        # Merge the features
        # merged_feat = torch.mean(torch.stack([feat_emote, feat_race]), dim=0)
        merged_feat = torch.cat((feat_emote, feat_race), dim=1)
        
        # Compute final logits
        final_logit = self.final_linear_layer(merged_feat)
        
        # Compute probabilities
        # emote_probs = torch.softmax(emote_logit, dim=-1)
        # race_probs = torch.softmax(race_logit, dim=-1)
        # final_probs = torch.softmax(final_logit, dim=-1)
                

        return emote_logit, race_logit, final_logit

class CulturalDualNetwork_ResVggFace_LateFusion(nn.Module):
    def __init__(self, num_classes, num_cultures):
        super().__init__()

        self.spatial_model_emote = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model_race = vgg_face_dag(weights_path='/home/hq/Documents/Weights/MicroEthnic/vgg_face_dag.pth')

        self.spatial_model_emote = nn.Sequential(*list(self.spatial_model_emote.children())[:-1])
        self.flatten = nn.Flatten()
        self.linear_layer_emotion = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        self.race_linear_layer = nn.Linear(in_features=576, out_features=num_cultures, bias=True)

        self.final_linear_layer = nn.Linear(in_features=1088, out_features=num_classes, bias=True)
        
        count_parameters(self.spatial_model_emote)
        count_parameters(self.spatial_model_race)

    
    def forward(self, image, image_rgb):
        # Extract features from both models
        feat_emote = self.spatial_model_emote(image)
        feat_emote = self.flatten(feat_emote)

        feat_race = self.spatial_model_race.forward_features(image_rgb)
        feat_race = self.spatial_model_race.norm_head(feat_race)

        emote_logit = self.linear_layer_emotion(feat_emote)
        race_logit = self.race_linear_layer(feat_race)
        

        # Merge the features
        # merged_feat = torch.mean(torch.stack([feat_emote, feat_race]), dim=0)
        merged_feat = torch.cat((feat_emote, feat_race), dim=1)
        
        # Compute final logits
        final_logit = self.final_linear_layer(merged_feat)

                

        return emote_logit, race_logit, final_logit





class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38

def vgg_face_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    # Get all parameter names
    all_params = list(model.named_parameters())
    # Freeze all layers except the last 14
    for name, param in all_params[:-7]:
        param.requires_grad = False

    return model

# a = vgg_face_dag(weights_path='/home/hq/Documents/Weights/MicroEthnic/vgg_face_dag.pth')
# # swin_v2 = swin_b(weights='IMAGENET1K_V1')
# a = 1

# a = CulturalDualNetwork_ResVggFace_LateFusion(num_classes=3, num_cultures=2)
# a = 1

# res18_imagenet(num_classes=3)


# res18_imagenet(num_classes=3)
# CulturalDualNetwork_TinyViT_LateFusion(num_classes=3, num_cultures=2)

# processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
# model = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224")
# a = 1