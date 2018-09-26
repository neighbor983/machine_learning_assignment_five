import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cost_run_plot(cost, runs, title, filename):
    plt.plot(runs, cost);
    plt.xlabel('Runs');
    plt.ylabel('Cost');
    plt.title(title);
    plt.savefig(filename);
    plt.close();