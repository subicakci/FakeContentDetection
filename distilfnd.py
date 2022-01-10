from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

import torch
from torch import nn
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertModel

CLASS_NAMES = ["True", "Satire / Parody", "False Conn.", "Impost. Content", "Man. Content", "Mis. Content"]

class DistilFND(nn.Module):

    def __init__(self, num_classes):
        super(DistilFND, self).__init__()
        self.title_module = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.image_module = models.resnet34(pretrained="imagenet")
        self.comment_module = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.drop = nn.Dropout(p=0.3)

        self.fc_title = nn.Linear(in_features=self.title_module.config.hidden_size, out_features=num_classes, bias=True)
        self.fc_comment = nn.Linear(in_features=self.comment_module.config.hidden_size, out_features=num_classes,
                                    bias=True)
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, title_input_ids, title_attention_mask, image, cm_input_ids, cm_attention_mask):
        title_last_hidden_states = self.title_module(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
            return_dict=False
        )
        title_pooled_output = title_last_hidden_states[0][:, 0, :]
        title_pooled_output = self.drop(title_pooled_output)

        image_output = self.image_module(image)
        image_output = self.drop(image_output)

        cm_last_hidden_states = self.comment_module(
            input_ids=cm_input_ids,
            attention_mask=cm_attention_mask,
            return_dict=False
        )

        cm_pooled_output = cm_last_hidden_states[0][:, 0, :]
        cm_pooled_output = self.drop(cm_pooled_output)

        title_condensed = self.fc_title(title_pooled_output)
        image_condensed = self.fc_image(image_output)
        cm_condensed = self.fc_comment(cm_pooled_output)

        fusion = torch.maximum(title_condensed, image_condensed)
        fusion = torch.add(fusion, cm_condensed)

        return self.softmax(fusion)

    def load_model(self):

        distilFND = DistilFND(len(CLASS_NAMES))
        distilFND.load_state_dict(torch.load("models/distilfnd.pth", map_location=torch.device("cpu")))

        return distilFND

    def tokenize(self, post_title, post_comments):

        title_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        comment_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

        title_encoding = title_tokenizer.encode_plus(
            post_title,
            max_length=80,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        post_comments = ""
        comment_encoding = comment_tokenizer.encode_plus(
            post_comments,
            max_length=80,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        title_input_ids = title_encoding["input_ids"]
        title_attention_mask = comment_encoding["attention_mask"]
        comment_input_ids = comment_encoding["input_ids"]
        comment_attention_mask = comment_encoding["attention_mask"]

        return title_input_ids, title_attention_mask, comment_input_ids, comment_attention_mask

    def process_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.255]
            )
        ])

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = transform(image)
        image = torch.unsqueeze(image, 0)

        return image

