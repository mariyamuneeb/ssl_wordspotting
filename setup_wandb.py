from google.colab import drive
import json
drive.mount('/content/drive')
GDRIVE_ROOT = '/content/drive/MyDrive'
PATH = f'{GDRIVE_ROOT}/Mariyah_Phd/secrets'
PATHW = f'{PATH}/wandb.json'


def get_wandb_key():
    with open(PATHW) as jsonfile:
        kdata = json.load(jsonfile)
    key = kdata['KEY']
    return key