import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class NerfModel(nn.Module):
  def __init__(self, embedding_dim_pos=10,embedding_dim_direction=4,hidden_dim=128):
    super(NerfModel, self).__init__()

    self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),)
    self.block2 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim+1),)
    self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + 3 + hidden_dim,hidden_dim//2),nn.ReLU(),)
    self.block4 = nn.Sequential(nn.Linear(hidden_dim//2,3),nn.Sigmoid(),)
    self.embedding_dim_direction = embedding_dim_direction
    self.embedding_dim_pos = embedding_dim_pos
    self.relu = nn.ReLU()

  def positional_encoding(self, x, L):
    out = [x]
    for j in range(L):
      out.append(torch.sin(2**j*x))
      out.append(torch.cos(2**j*x))
    return torch.cat(out,dim=1)

  def forward(self, o, d):
    emb_d = self.positional_encoding(d, self.embedding_dim_direction)
    emb_x = self.positional_encoding(o, self.embedding_dim_pos)
    h = self.block1(emb_x)
    tmp = self.block2(torch.cat((h,emb_d),dim=1))
    h, sigma = tmp[: , :-1], self.relu(tmp[:,-1])
    h = self.block3(torch.cat((h,emb_d),dim=1))
    c = self.block4(h)
    return c,sigma

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas,1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0],1),device = alphas.device),
                      accumulated_transmittance[:,:-1]),
                      dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    #sampling the points using stratified sampling
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins,device = device).expand(ray_origins.shape[0],nb_bins)
    mid = (t[:,:-1]+t[:,1:])/2
    lower = torch.cat((t[:,:1],mid),-1)
    upper = torch.cat((mid,t[:,-1:]),-1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper-lower)*u

    delta = torch.cat((t[:,1:]-t[:,:-1],torch.tensor([1e10],device=device).expand(ray_origins.shape[0],1)),-1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions = ray_directions.expand(nb_bins,ray_directions.shape[0],3).transpose(0,1)

    colours,sigma = nerf_model(x.reshape(-1,3),ray_directions.reshape(-1,3))
    colours = colours.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma*delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colours).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1) # regularization for white background
    return (c + 1 - weight_sum.unsqueeze(-1))

def train(nerf_model, testing_dataset,optimizer, data_loader, scheduler, device = 'cpu', hn = 0, hf = 1,nb_epochs = int(1e5), nb_bins = 192, H=400, W=400):
    training_loss = []
    for _ in tqdm(range(nb_epochs)):
      print('loop is running')
      for batch in tqdm(data_loader):
        ray_origins = batch[:,:3].to(device)
        ray_directions = batch[:,3:6].to(device)
        ground_truth_px_values = batch[:,6:].to(device)

        regenerated_px_values = render_rays(nerf_model,ray_origins,ray_directions,hn=hn,hf=hf,nb_bins = nb_bins)
        loss = ((ground_truth_px_values-regenerated_px_values) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        # break
      scheduler.step()
      torch.save(model.state_dict(), f'nerf/checkpoints/model_{_}.pt')
      
      for img_index in range(200):
        test(nerf_model, device, hn, hf, testing_dataset, img_index= img_index, nb_bins=nb_bins, H=H,W=W)
        # print("Test Done")

    return training_loss

@torch.no_grad()
def test(model, device, hn, hf, dataset, chunk_size = 10,img_index = 0, nb_bins=192, H=400,W=400):
  ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
  ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, 3:6]
  data = []
  # print(chunk_size)
  for i in tqdm(range(int(np.ceil(H/(chunk_size+1e-8))))):
    ray_origins_ = ray_origins[i * W * chunk_size:(i+1) * W * chunk_size].to(device)
    ray_directions_ = ray_directions[i * W * chunk_size:(i+1) * W * chunk_size].to(device)
    regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
    data.append(regenerated_px_values)

  img = torch.cat(data).cpu().numpy().reshape(H, W, 3)
  plt.figure()
  plt.imshow(img)
  plt.savefig(f"novel_view/img_{img_index}.png",bbox_inches = 'tight')
  
  

if __name__ == '__main__':
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('nerf/training_data.pkl',allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('nerf/testing_data.pkl',allow_pickle=True ))
    model = NerfModel(hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    print("CUDA Available:", torch.cuda.is_available())
    print("Model Device:", next(model.parameters()).device)
    data_loader = DataLoader(training_dataset, batch_size=1536, shuffle=True)    
    train(model, testing_dataset, optimizer, data_loader, scheduler, device = device, hn = 2, hf = 6,nb_epochs = 12, nb_bins = 192, H=400, W=400)