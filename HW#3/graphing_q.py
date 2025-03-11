from util.graphing import graph_loss_pb
from matplotlib import pyplot
import os
from pathlib import Path
import time

def q_1_graphing():
    acc =        [0.296, 0.618,  0.596,  0.622,   0.622,  0.604,   0.628,   0.622, 0.624]
    loss_epoch = [0.614, 0.0136, 0.0692, 0.00103, 0.0319, 0.00433, 0.000461,0.0043,4.29e-5]
    fig, ax = pyplot.subplots(1,2,figsize=(20,10))

    graph_loss_pb(ax[0], acc, x_label="Epoch", y_label="Acc", title="Classification lr=3e-5")
    graph_loss_pb(ax[1], loss_epoch, x_label="Epoch", y_label="Loss", title="Classification lr=3e-5")
    pyplot.savefig(Path(os.path.abspath(__file__)).parent / f"classification_graph_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png")

def q_2_graphing():
    acc = [0.02, 0.228, 0.276, 0.266, 0.212, 0.202, 0.192, 0.16, 0.184, 0.184]
    loss_epoch = [2.4606455078125, 1.9158158168792725, 1.628836593747139, 1.3949511609077454, 1.179824093580246, 1.0005292154550551, 0.8367087923884392, 0.694836210668087, 0.6075680565536022, 0.5008196629583835]

    fig, ax = pyplot.subplots(1,2,figsize=(20,10))

    graph_loss_pb(ax[0], acc, x_label="Epoch", y_label="Acc", title="Generative lr=3e-5")
    graph_loss_pb(ax[1], loss_epoch, x_label="Epoch", y_label="Loss", title="Generative lr=3e-5")
    pyplot.savefig(Path(os.path.abspath(__file__)).parent / f"generation_graph_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png")

q_1_graphing()
q_2_graphing()