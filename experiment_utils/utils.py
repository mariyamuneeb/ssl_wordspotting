import wandb
import torch


def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader:
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


def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    # reconstruct image
    # calculate mAP
    vae.eval()
    val_loss = 0.0
    # image re-construction
    with torch.no_grad():  # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
            val_loss += loss.item()
    val_loss_ave = val_loss / len(dataloader.dataset)
    wandb.log({"val_loss": val_loss_ave})
    # todo calculate mAP
    return val_loss_ave



def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss, path):
    pass


def load_checkpoint(path):
    return torch.load(path)
