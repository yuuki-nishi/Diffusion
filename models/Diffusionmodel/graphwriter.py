import matplotlib.pyplot as plt
import makeconfig

def write(losslist,stdes,config : makeconfig.Myconfig):
    x_values = range(len(losslist))
    plt.plot(x_values, losslist, color="k")
    plt.savefig(config.lossgraphpath)
    #plt.show()
    #plt.ctf()
    x_values = range(len(stdes))
    plt.plot(x_values, stdes, color="k")
    plt.savefig(config.stdplotpath)
    return