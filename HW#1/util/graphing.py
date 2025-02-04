import matplotlib.pyplot as pyplot

def graph_loss_direct(train_loss:list[int],x_label: str, y_label:str, lr:float):
    pyplot.plot([i for i in range(len(train_loss))],train_loss)
    pyplot.xlabel(y_label)
    pyplot.title(f"lr = {lr}")
    pyplot.ylabel(x_label)

def graph_loss_pb(ax: pyplot.Axes, train_loss:list[int],x_label: str, y_label:str, lr:float,y_max:int):
    ax.plot([i for i in range(len(train_loss))],train_loss)
    ax.set_xlabel(y_label)
    ax.set_title(f"lr = {lr}")
    ax.set_ylabel(x_label)
    ax.set_ybound(0,y_max)



def graph_loss_each_epoch_pb(ax: pyplot.Axes, train_loss_each_epoch:list[list[int]],x_label: str, y_label:str, lr:float):
    for i in range(len(train_loss_each_epoch)):
        ax.plot([i for i in range(len(train_loss_each_epoch[i]))],train_loss_each_epoch[i], label=f'Epoch {i}')
    ax.set_xlabel(y_label)
    ax.set_title(f"lr = {lr}")
    ax.set_ylabel(x_label)