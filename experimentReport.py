import matplotlib.pyplot as plt

def experimentReport(summary):
# plt.plot(summary.history["acc"])
# plt.plot(summary.history['val_acc'])
    plt.plot(summary.history['loss'])
    plt.plot(summary.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss","Validation Loss"])
    plt.show()
    plt.savefig('chart loss.png')