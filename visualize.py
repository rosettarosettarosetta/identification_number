import torch
import matplotlib.pyplot as plt
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量
            ax.imshow(img.numpy())
        else:
            # PIL图⽚
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    return list(map(str, labels))
