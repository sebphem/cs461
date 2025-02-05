import matplotlib.pyplot as pyplot

def graph_loss_direct(train_loss:list[int],x_label: str, y_label:str, title:str):
    pyplot.plot([i for i in range(len(train_loss))],train_loss)
    pyplot.xlabel(x_label)
    pyplot.title(title)
    pyplot.ylabel(y_label)

def graph_loss_pb(ax: pyplot.Axes, train_loss:list[int],x_label: str, y_label:str, title:str,y_max:int):
    ax.plot([i for i in range(len(train_loss))],train_loss)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_ybound(0,y_max)

def graph_loss_each_epoch_pb(ax: pyplot.Axes, train_loss_each_epoch:list[list[int]], x_label: str, y_label:str, title:str):
    for i in range(len(train_loss_each_epoch)):
        ax.plot([i for i in range(len(train_loss_each_epoch[i]))],train_loss_each_epoch[i], label=f'Epoch {i}')
    ax.set_xlabel(x_label)
    ax.set_xticks([i for i in range(len(train_loss_each_epoch[0]))])
    ax.set_title(title)
    ax.set_ylabel(y_label)

def graph_loss_pb(ax: pyplot.Axes, train_loss_each_epoch:list[list[int]], x_label: str, y_label:str, title:str):
    ax.plot([i for i in range(len(train_loss_each_epoch))],train_loss_each_epoch)
    ax.set_xlabel(x_label)
    ax.set_xticks([i for i in range(len(train_loss_each_epoch))])
    ax.set_title(title)
    ax.set_ylabel(y_label)