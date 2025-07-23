import torch
import pandas as pd
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
            
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description
run_number = input("Enter the Run Number: ")

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

print(logs_save_dir)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

##Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")


# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"supervised_seed_{SEED}", "saved_models"))

chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)

pretrained_dict = chkpoint["model_state_dict"]


model_dict = model.state_dict()
del_list = ['logits']
pretrained_dict_copy = pretrained_dict.copy()
for i in pretrained_dict_copy.keys():
  for j in del_list:
    if j in i:
      del pretrained_dict[i]
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

## Skiping to model evaluate

print('Testing Model')

outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)

total_loss, total_acc, pred_labels, true_labels = outs


pred_labels = np.array(pred_labels)
true_labels = np.array(true_labels)

os.makedirs(os.path.join(experiment_log_dir, "Test_Data"), exist_ok=True)

##Making the classification report 
r = classification_report(true_labels, pred_labels, digits = 6, output_dict=True)
df = pd.DataFrame(r)
df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
df["accuracy"] = accuracy_score(true_labels, pred_labels)
df = df * 100

##Saving Classification Report 
exp_name = os.path.split(os.path.dirname(experiment_log_dir))[-1]
training_mode = os.path.basename(experiment_log_dir)
file_name = f"classification_report{run_number}.xlsx"

home_path = args.home_path

report_Save_path = os.path.join(home_path, experiment_log_dir, "Test_Data", file_name)

df.to_excel(report_Save_path)

##Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")

##Saving Confusion Matrix

plt.savefig(os.path.join(experiment_log_dir,"Test_Data",f"confusion_matrix{run_number}.png"))

plt.show()