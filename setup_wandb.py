from google.colab import drive
import json
import wandb

from data_utils.utils import connect_to_gdrive

drive.mount('/content/drive')
GDRIVE_ROOT = '/content/drive/MyDrive'


def get_wandb_key():
    GDRIVE_ROOT = connect_to_gdrive()
    PATH = f'{GDRIVE_ROOT}/Mariyah_Phd/secrets'
    PATHW = f'{PATH}/wandb.json'
    with open(PATHW) as jsonfile:
        kdata = json.load(jsonfile)
    key = kdata['KEY']
    return key


def wandb_login():
    wandb.login(key=get_wandb_key())
