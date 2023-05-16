import logging

import wandb
import torch
import numpy as np
from tqdm import tqdm

from experiment_utils.metrics import map_from_feature_matrix


def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for _, x, _ in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    train_loss_ave = train_loss / len(dataloader.dataset)
    wandb.log({"train_loss": train_loss_ave})
    return train_loss_ave


def evaluate_model(model, dataset_loader, args):
    logger = logging.getLogger('Experiment::test')
    # set the CNN in eval mode
    model.eval()
    logger.info('Computing net output:')
    qry_ids = []  # np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    for sample_idx, (img_id, img, label, class_id, is_query) in enumerate(tqdm(dataset_loader)):
        if args.gpu_id is not None:
            # in one gpu!!
            word_img = img.cuda(args.gpu_id[0])
            # embedding = embedding.cuda(args.gpu_id[0])
            # word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
        word_img = torch.autograd.Variable(img)
        # embedding = torch.autograd.Variable(embedding)

        output = torch.sigmoid(model(word_img))
        # output = cnn(word_img)
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        # embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0, 0]
        if is_query[0] == 1:
            qry_ids.append(sample_idx)  # [sample_idx] = is_query[0]

    '''
    # find queries
    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]
    # remove stopwords if needed

    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                         test_features=outputs,
                                                         query_labels=qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric='cosine',
                                                         drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[ave_precs_qbe > 0]) * 100)

    # clean up -> set CNN in train mode again
    model.train()


def evaluate_qbe(model, latent_dim, device, dataloader):
    embedding_size = latent_dim
    embeddings = np.zeros((len(dataloader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataloader), embedding_size), dtype=np.float32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    for sample_idx, (img_id, img, transcript, class_id) in enumerate(dataloader):
        output = torch.sigmoid(vae.encoder(img))
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id


def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    # reconstruct image
    # calculate mAP
    vae.eval()
    val_loss = 0.0
    features_list = list()
    labels_list = list()
    # image re-construction & mAP

    with torch.no_grad():  # No need to track the gradients
        for _, img, labels, _ in dataloader:
            # Move tensor to the proper device
            img = img.to(device)
            # Encode data
            feature = vae.encoder(img)
            feature = torch.squeeze(feature)
            feature = feature.cpu().detach().numpy()
            features_list.append(feature)
            labels_list.append(list(labels))
            # Decode data
            img_hat = vae(img)
            loss = ((img - img_hat) ** 2).sum() + vae.encoder.kl
            val_loss += loss.item()
    feature_array = np.concatenate(features_list)
    labels_list = [item for sublist in labels_list for item in sublist]
    mean_ap, _ = map_from_feature_matrix(feature_array, labels_list, 'euclidean', True)
    val_loss_ave = val_loss / len(dataloader.dataset)
    wandb.log({"val_loss": val_loss_ave})
    wandb.log({"mAP": mean_ap})
    # todo calculate mAP
    return val_loss_ave, mean_ap


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss, path):
    pass


def load_checkpoint(path):
    return torch.load(path)
