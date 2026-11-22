# Model architecture adapted from the authors' released DUPGT code.
import torch
import torch.nn.functional as F


# Store a small learned vector for each user and item.

class LookupEmbedding(torch.nn.Module):
  def __init__(self, uid_all, iid_all, emb_dim):
    super().__init__()
    self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
    self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

  def forward(self, x):
    
    # x contains a user ID and an item ID for each rating.

    uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
    iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
    emb = torch.cat([uid_emb, iid_emb], dim=1)
    return emb


# Turn a Book history into a matrix that transforms a user vector.

class DeepMetaNet(torch.nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, 7), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(7, 4), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(4, 2), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(2, 1, False))
    self.event_softmax = torch.nn.Softmax(dim=1)
    self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, 25), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(25, 50), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(50, 75), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(75, emb_dim * emb_dim))

  def forward(self, emb_fea, seq_index):

    # Ignore padded item ID 0 when assigning attention weights.

    mask = (seq_index == 0).float()
    event_K = self.event_K(emb_fea)
    t = event_K - torch.unsqueeze(mask, 2) * 1e8
    att = self.event_softmax(t)

    # Combine the important Books, then predict the transformation matrix.

    his_fea = torch.sum(att * emb_fea, 1)
    output = self.decoder(his_fea)
    return output.squeeze(1)
    

# Choose how much to use the high-rating and low-rating Book paths.

class BlockWiseGatingMLP(torch.nn.Module):
  def __init__(
    self,
    emb_dim: int,
    proj_dim: int = 128,              
    hidden_dim: int = 384,            
  ):
    super().__init__()

    d, d2 = emb_dim, emb_dim * emb_dim
    self.input_dims = {
      "H": d2,   # high-rating matrix
      "L": d2,   # low-rating matrix
      "U": d,    # Book user vector
      "I": d,    # candidate Movie vector
      "D": d2    # difference between the matrices
    }

    self.proj = torch.nn.ModuleDict()
    for name, in_dim in self.input_dims.items():
      self.proj[name] = torch.nn.Sequential(
        torch.nn.Linear(in_dim, proj_dim * 2),
        torch.nn.GLU(dim=-1)
      )

    concat_dim = len(self.input_dims) * proj_dim
    self.norm = torch.nn.LayerNorm(concat_dim)
    self.net = torch.nn.Sequential(torch.nn.Linear(concat_dim, hidden_dim),torch.nn.ELU(0.8),torch.nn.Dropout(0.2),
                                    torch.nn.Linear(hidden_dim, hidden_dim),torch.nn.ELU(0.8),torch.nn.Dropout(0.2),
                                    torch.nn.Linear(hidden_dim, hidden_dim),torch.nn.ELU(0.8),
                                    torch.nn.Linear(hidden_dim, 1)
    )

  def forward(self, bridgingH, bridgingL, user_emb, item_emb):

    # The gate depends on both Book paths, the user, and the Movie.

    diff = bridgingH - bridgingL

    block_outs = []
    for name, tensor in zip(
      ["H", "L", "U", "I", "D"],
      [bridgingH, bridgingL, user_emb, item_emb, diff]
    ):
      h = self.proj[name](tensor)
      block_outs.append(h)

    cat_fea = self.norm(torch.cat(block_outs, dim=1))
    alpha_logit = self.net(cat_fea).squeeze(1)

    # Sigmoid keeps the mixing weight between 0 and 1.

    alpha = torch.sigmoid(alpha_logit)
    return alpha


# Build separate transformations from liked and less-liked Books.

class DoubleBridging_UserItemGate(torch.nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.bridging_high = DeepMetaNet(emb_dim)
    self.bridging_low  = DeepMetaNet(emb_dim)
    self.gating_mlp = BlockWiseGatingMLP(emb_dim=emb_dim, hidden_dim=384)

  def forward(self, emb_fea, seq_index, user_emb, item_emb):

    # The first 20 items are high ratings; the next 20 are low ratings.

    bridgingH = self.bridging_high(emb_fea[:,0:20],  seq_index[:,0:20])
    bridgingL = self.bridging_low( emb_fea[:,20:40], seq_index[:,20:40])
    alpha = self.gating_mlp(bridgingH, bridgingL, user_emb, item_emb)
    alpha_expand = alpha.unsqueeze(1)

    # Blend the two matrices using this user-Movie-specific weight.

    bridging = alpha_expand*bridgingH + (1 - alpha_expand)*bridgingL
    return bridging


# Keep Book and Movie embeddings separate, then transfer between them.

class Model(torch.nn.Module):
  def __init__(self, uid_all, iid_all, num_fields, emb_dim):
    super().__init__()
    self.num_fields = num_fields
    self.emb_dim = emb_dim
    self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
    self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
    self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
    self.dupgt = DoubleBridging_UserItemGate(emb_dim)
    self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

  def forward(self, x, stage):
    if stage == 'train_src':

      # Learn Book vectors from observed Book ratings.

      emb = self.src_model(x)
      out = torch.sum(emb[:,0,:]*emb[:,1,:], dim=1)
      return out

    elif stage in ['train_tgt','test_tgt']:

      # Learn Movie vectors from observed Movie ratings.

      emb = self.tgt_model(x)
      out = torch.sum(emb[:,0,:]*emb[:,1,:], dim=1)
      return out

    elif stage in ['train_aug','test_aug']:

      # Extra embedding branch supplied by the authors' code.

      emb = self.aug_model(x)
      out = torch.sum(emb[:,0,:]*emb[:,1,:], dim=1)
      return out
    
    elif stage == 'train_map':

      # Fit a simple linear Book-user to Movie-user baseline.

      src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
      src_emb_mapped = self.mapping(src_emb)
      tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
      return src_emb_mapped, tgt_emb

    elif stage == 'test_map':

      # Score a Movie using the linearly mapped Book-user vector.

      uid_emb = self.mapping(self.src_model.uid_embedding(x[:,0].unsqueeze(1)).squeeze())
      emb = self.tgt_model(x)
      emb[:,0,:] = uid_emb
      out = torch.sum(emb[:,0,:]*emb[:,1,:], dim=1)
      return out

    elif stage in ['train_meta','test_meta']:

      # DUPGT uses a Book user, a Movie item, and 40 Book-history slots.

      uid = x[:,0]
      iid = x[:,1]
      seq_items = x[:, 2:42]  # item history(high + low) in source domain 

      # Start with the user's learned Book vector.

      uid_emb_src = self.src_model.uid_embedding(uid.unsqueeze(1))
      user_emb = uid_emb_src.squeeze(1)

      # Get the candidate Movie's learned vector.

      iid_emb_tgt = self.tgt_model.iid_embedding(iid.unsqueeze(1))
      item_emb = iid_emb_tgt.squeeze(1)

      # Build a Book-to-Movie transformation from their Book history.

      emb_fea = self.src_model.iid_embedding(seq_items)
      bridging = self.dupgt(emb_fea, seq_items, user_emb, item_emb).view(-1, self.emb_dim, self.emb_dim)

      # Apply that transformation to the Book user vector.

      uid_emb_tgt = torch.bmm(uid_emb_src, bridging)

      # A vector dot product gives the predicted Movie rating.

      emb_cat = torch.cat([uid_emb_tgt, iid_emb_tgt], dim=1)
      rating = torch.sum(emb_cat[:,0,:]*emb_cat[:,1,:], dim=1)
      return rating

    else:
      raise ValueError(f"Unknown stage: {stage}")
