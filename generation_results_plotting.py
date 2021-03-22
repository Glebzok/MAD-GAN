import matplotlib.pyplot as plt

def plot_1d_generated(fake, settings, epoch):
  fig, ax = plt.subplots(settings['num_signals'], 1, figsize=(10, 10))

  for i in range(settings['num_signals']):
    ax[i].plot(fake[0, :, i])
    # ax[i].axis('off')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
  plt.tight_layout()
  plt.savefig('./generation_results/{}_{}.png'.format(settings['data'].__name__, epoch))
  plt.close(fig)

def plot_2d_generated(fake, settings, epoch):
  n_images = min(6, settings['seq_length'])
  step = settings['seq_length'] // n_images
  fig, ax = plt.subplots(1, n_images, figsize=(15, 1))

  for i in range(n_images):
    ax[i].imshow(fake[0, step*i])
    ax[i].axis('off')

  plt.savefig('./generation_results/{}_{}.png'.format(settings['data'].__name__, epoch))
  plt.close(fig)
