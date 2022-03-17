# This code is mainly referencing to:
# https://github.com/airalcorn2/pytorch-nerf

import torch
import torch.nn as nn

# Full NeRF MLP
class NeRFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of encoding functions for positions. See Section 5.1.
        self.L_pos = 10
        # Number of encoding functions for viewing directions. See Section 5.1.
        self.L_dir = 4
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir

        in_feats = pos_enc_feats
        net_width = 256
        early_mlp_layers = 5
        early_mlp = []
        for _ in range(early_mlp_layers):
            early_mlp.append(nn.Linear(in_feats, net_width))
            early_mlp.append(nn.ReLU())
            in_feats = net_width

        self.early_mlp = nn.Sequential(*early_mlp)

        in_feats = pos_enc_feats + net_width
        late_mlp_layers = 3
        late_mlp = []
        for _ in range(late_mlp_layers):
            late_mlp.append(nn.Linear(in_feats, net_width))
            late_mlp.append(nn.ReLU())
            in_feats = net_width

        self.late_mlp = nn.Sequential(*late_mlp)
        self.sigma_layer = nn.Linear(net_width, net_width + 1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(dir_enc_feats + net_width, net_width // 2), nn.ReLU()
        )
        self.final_layer = nn.Sequential(nn.Linear(net_width // 2, 3), nn.Sigmoid())

    def forward(self, xs, ds):
        # Encode the inputs. See Equation (4) in Section 5.1.
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2**l_pos * torch.pi * xs))
            xs_encoded.append(torch.cos(2**l_pos * torch.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2**l_dir * torch.pi * ds))
            ds_encoded.append(torch.cos(2**l_dir * torch.pi * ds))

        ds_encoded = torch.cat(ds_encoded, dim=-1)

        # Use the network to predict colors (c_is) and volume densities (sigma_is) for
        # 3D points (xs) along rays given the viewing directions (ds) of the rays. See
        # Section 3 and Figure 7 in the Supplementary Materials.
        outputs = self.early_mlp(xs_encoded)
        outputs = self.late_mlp(torch.cat([xs_encoded, outputs], dim=-1))
        outputs = self.sigma_layer(outputs)
        sigma_is = torch.relu(outputs[:, 0])
        outputs = self.pre_final_layer(torch.cat([ds_encoded, outputs[:, 1:]], dim=-1))
        c_is = self.final_layer(outputs)
        return {"c_is": c_is, "sigma_is": sigma_is}


# NeRF MLP without positional encoding
class VanillaNeRFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        in_feats = 3
        net_width = 256
        early_mlp_layers = 5
        early_mlp = []
        for _ in range(early_mlp_layers):
            early_mlp.append(nn.Linear(in_feats, net_width))
            early_mlp.append(nn.ReLU())
            in_feats = net_width

        self.early_mlp = nn.Sequential(*early_mlp)

        in_feats = 3 + net_width
        late_mlp_layers = 3
        late_mlp = []
        for _ in range(late_mlp_layers):
            late_mlp.append(nn.Linear(in_feats, net_width))
            late_mlp.append(nn.ReLU())
            in_feats = net_width

        self.late_mlp = nn.Sequential(*late_mlp)
        self.sigma_layer = nn.Linear(net_width, net_width + 1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(3 + net_width, net_width // 2), nn.ReLU()
        )
        self.final_layer = nn.Sequential(nn.Linear(net_width // 2, 3), nn.Sigmoid())

    def forward(self, xs, ds):
        # Use the network to predict colors (c_is) and volume densities (sigma_is) for
        # 3D points (xs) along rays given the viewing directions (ds) of the rays. See
        # Section 3 and Figure 7 in the Supplementary Materials.
        outputs = self.early_mlp(xs)
        outputs = self.late_mlp(torch.cat([xs, outputs], dim=-1))
        outputs = self.sigma_layer(outputs)
        sigma_is = torch.relu(outputs[:, 0])
        outputs = self.pre_final_layer(torch.cat([ds, outputs[:, 1:]], dim=-1))
        c_is = self.final_layer(outputs)
        return {"c_is": c_is, "sigma_is": sigma_is}


def get_model(device, positional_encoding=True):
    if positional_encoding:
        model = NeRFMLP().to(device)
    else:
        model = VanillaNeRFMLP().to(device)
    return model