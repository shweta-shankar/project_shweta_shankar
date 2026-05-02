# The required interface aliases as specified in the PDF format guidelines.
from model import get_mobilenet_model as TheModel
from train import train_model as the_trainer
from predict import classify_images as the_predictor
from dataset import ASLDataset as TheDataset
from dataset import get_dataloaders as the_dataloader
from config import batch_size as the_batch_size
from config import number_of_epochs as total_epochs
