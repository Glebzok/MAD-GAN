import numpy as np
import torch

def covariance_similarity(tensor1, tensor2):
  mean1 = tensor1.mean(axis=-1)
  mean1_broadcasted = torch.broadcast_tensors(tensor1.T, mean1.T)[1].T
  tensor1_center = tensor1 - mean1_broadcasted
  std1 = tensor1.std(axis=2)
  std1_broadcasted = torch.broadcast_tensors(tensor1.T, std1.T)[1].T

  mean2 = tensor2.mean(axis=-1)
  mean2_broadcasted = torch.broadcast_tensors(tensor2.T, mean2.T)[1].T
  tensor2_center = tensor2 - mean2_broadcasted
  std2 = tensor2.std(axis=2)
  std2_broadcasted = torch.broadcast_tensors(tensor2.T, std2.T)[1].T

  std_broadcasted = std1_broadcasted * std2_broadcasted
  res = tensor1_center * tensor2_center / std_broadcasted
  res = res.mean(axis=2)

  return res

def reconstruct(data, generator, discriminator, optimizer, similarity, latent_space_dim, device, tol=1e-3, max_iter=100):

  x = data.to(device)

  z = torch.randn(x.shape[0], x.shape[1], latent_space_dim, device=device, requires_grad=True)
  optimizer = optimizer([z])
  g_z = generator(z)

  reconstruction_error = 1 - similarity(x, g_z)

  prev_error = reconstruction_error.max().detach().cpu().numpy()

  for _ in range(max_iter):
    reconstruction_error.mean().backward()
    optimizer.step()

    g_z = generator(z)
    reconstruction_error = 1 - similarity(x, g_z)

    z.grad.zero_()

    if np.abs(prev_error - reconstruction_error.max().detach().cpu().numpy()) < tol:
      break
    else:
      prev_error = reconstruction_error.max().detach().cpu().numpy()

  return z, reconstruction_error.detach().cpu().numpy()


def detect_anomalies(data, generator, discriminator, optimizer, similarity, device, fake_label, latent_space_dim, lmbda, tau, tol=1e-3, max_iter=100):
  data = data.to(device)

  _, reconstruction_loss = reconstruct(data, generator, discriminator, optimizer, similarity, latent_space_dim, device, tol, max_iter)

  disciminator_output = discriminator(data).detach().cpu().numpy()[:, :, 0]
  discrimination_loss = fake_label * disciminator_output + (1 - fake_label) * (1 - disciminator_output)
  anomaly_detection_loss = lmbda * reconstruction_loss + (1 - lmbda) * discrimination_loss

  anomaly_detected = (anomaly_detection_loss > tau).astype('int')

  return anomaly_detection_loss, anomaly_detected