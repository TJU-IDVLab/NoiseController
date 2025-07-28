import torch
import torch.nn as nn
import torch.nn.functional as F  

def noise_slide_window(total_w, total_impact, B_w_impact_loss=None, I_w_impact_loss=None, h=7, w=10, c=4, batch_size=2, residual_proportion=(1./5.)):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    k = total_impact.shape[-1]
    video_len = total_w.shape[-1]
    visual_angle = total_w.shape[1]

    input_B = torch.randn(batch_size, h, w, c, visual_angle, device=device)
    input_I = torch.randn(batch_size, h, w, c, visual_angle, device=device)
    
    B_noise_group = []
    I_noise_group = []

    total_w = total_w.permute(0, -1, 1, 2)

    total_B_impact = total_impact[0]
    total_I_impact = total_impact[1]
    total_B_impact = total_B_impact.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    total_I_impact = total_I_impact.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    new_B_noise = input_B
    new_I_noise = input_I

    B_noise_group.append(new_B_noise.clone())
    I_noise_group.append(new_I_noise.clone())
    
    for i in range(1, video_len+1):

        B_residual = torch.randn((batch_size, h, w, c, visual_angle), device=device) * torch.sqrt(torch.tensor(residual_proportion, device=device))
        I_residual = torch.randn((batch_size, h, w, c, visual_angle), device=device) * torch.sqrt(torch.tensor(residual_proportion, device=device))

        if i < k:
            # if index of generated noise is less than the length of windows
            sub_B_noise = B_noise_group[:i]
            sub_I_noise = I_noise_group[:i]

            total_noise = torch.stack([torch.stack(sub_B_noise, dim=-2), torch.stack(sub_I_noise, dim=-2)], dim=4).unsqueeze(-2)
            
            noise_w = torch.matmul(total_noise, total_w[:, :i, :, :])

            # calculate new noise
            new_B_noise = torch.sum(noise_w * total_B_impact[..., :i, :, :], dim=(4, 5, 6)) + B_residual
            new_I_noise = torch.sum(noise_w * total_I_impact[..., :i, :, :], dim=(4, 5, 6)) + I_residual

            B_noise_group.append(new_B_noise.clone())
            I_noise_group.append(new_I_noise.clone())

        else:
            total_noise = torch.stack([torch.stack(B_noise_group[i-k:i], dim=-2), torch.stack(I_noise_group[i-k:i], dim=-2)], dim=4).unsqueeze(-2)

            noise_w = torch.matmul(total_noise, total_w[:, i-k:i, :, :])

            new_B_noise = torch.sum(noise_w * total_B_impact, dim=(4, 5, 6)) + B_residual
            new_I_noise = torch.sum(noise_w * total_I_impact, dim=(4, 5, 6)) + I_residual

            B_noise_group.append(new_B_noise.clone())
            I_noise_group.append(new_I_noise.clone())

    B_noise_group = torch.stack(B_noise_group, dim=0)
    I_noise_group = torch.stack(I_noise_group, dim=0)

    B_noise_group =  B_noise_group.permute(1, 0, 5, 4, 2, 3)
    I_noise_group =  I_noise_group.permute(1, 0, 5, 4, 2, 3)

    return B_noise_group.squeeze(0), I_noise_group.squeeze(0)