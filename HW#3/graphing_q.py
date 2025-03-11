from util.graphing import graph_loss_pb
from matplotlib import pyplot
import os
from pathlib import Path
import time

def q_1_graphing():
    acc = [0.272, 0.24, 0.242, 0.266, 0.26, 0.282, 0.276, 0.276, 0.276, 0.276, 0.278, 0.284, 0.276, 0.284, 0.276, 0.286, 0.296, 0.265, 0.291, 0.290]
    loss_epoch = [2.03266409305393, 1.74076873469742661, 1.619681632778307, 1.4376935987933854, 1.228973655496241, 1.1622839367634046, 1.0366703647351303, 0.913449878016826,  0.8413269959884744, 0.7201863376377713,
                0.8140738807682021, 0.6744639604238589, 0.5579897165555435, 0.5361260152739308, 0.7732945159965767, 0.528313176617913, 0.634298815366664, 0.4675421830380333,  0.627656611169931,  0.5050136752611045,                  ]
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