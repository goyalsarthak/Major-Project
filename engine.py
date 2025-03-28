from typing import Iterable
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import util.misc as utils
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_meandice
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import rescale_intensity
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
# from torchvision.transforms import functional as F
from PIL import Image
import math
import numpy as np
print = functools.partial(print, flush=True)

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name)


def preprocess_images(images, processor, device): 
    # print(images.shape, "################$#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", images.dtype,"\n\n")
    images = (images - images.min()) / (images.max() - images.min())
    images = images.clamp(0.0, 1.0)
    images = images.to(torch.float32)
    images = images.repeat(1, 3, 1, 1)
    # images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    # images = torch.from_numpy(images).to(device)
#     # Use SegformerImageProcessor to normalize, resize, and process the images
#     # It will automatically handle normalization, resizing, and converting to tensors
    # print(images.shape, "################$#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", images.dtype,"\n\n")
    resized_tensor = F.interpolate(images, size=(512, 512), mode='bicubic', align_corners=False)
    # processed_images = processor(images, return_tensors='pt', do_rescale = False, size = (512, 512)).to(device)
    return resized_tensor

def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10
    cur_iteration=0
    while True:
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
            for k,v in samples.items():
                if isinstance(samples[k],torch.Tensor):
                    samples[k]=v.to(device)
            cur_iteration+=1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

            img=samples['images']
            lbl=samples['labels']
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if cur_iteration>=warmup_iteration:
                print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration
        metric_logger.synchronize_between_processes()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1, grad_scaler=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        img = samples['images']
        lbl = samples['labels']

        if grad_scaler is None:
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss_dict = criterion.get_loss(pred,lbl)
                losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            grad_scaler.scale(losses).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cur_iteration+=1
        if cur_iteration>=max_iteration and max_iteration>0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration

def process_attention_maps(attention_maps, target_size=(192, 192)):
    """
    Process Segformer attention maps to extract, normalize, and resize them.
    
    Args:
        attention_maps (torch.Tensor): Attention maps from Segformer, shape (batch_size, num_heads, num_tokens, num_tokens).
        target_size (tuple): The target spatial size for resizing the attention map.
    
    Returns:
        torch.Tensor: Processed attention maps resized to the target image size.
    """
    # Get the last layer's attention map
    last_attention_map = attention_maps[-1]  # Shape: [batch_size, num_heads, num_tokens, num_tokens]
    
    # Average across heads
    attn_weights1 = last_attention_map.mean(dim=1)  # Shape: [batch_size, num_tokens, num_tokens]
    
    # Apply ReLU activation and reshape
    # attention_head = F.relu(attn_weights1)  # Ensure non-negative values
    
    # Reshape for interpolation
    attention_head = attn_weights1.unsqueeze(1)  # Add channel dimension
    
    # Resize to target image size
    attn_map_resized = F.interpolate(attention_head, size=target_size, mode='bicubic', align_corners=False)
    normalized_map = rescale_intensity(attn_map_resized)
    
    return normalized_map

# def aggregate_attention_maps(attention_maps):
#     """
#     Extract and process attention maps from SegFormer, preserving batch size.
    
#     Args:
#         attention_maps (tuple): Tuple of attention tensors from SegFormer
#         target_size (tuple): Desired output size for attention map
    
#     Returns:
#         torch.Tensor: Normalized attention maps of shape [batch_size, 1, 512, 512]
#     """
#     last_attention_map = attention_maps[-1]
#     attn_weights1 = last_attention_map.mean(dim=1) 
#     attention_map = attention_maps[-1]
#     attention_map=torch.mean(attention_map.unsqueeze(1), dim=1)
#     attention_map = torch.mean(attention_map, dim=1)  # Shape: (32, 256, 256)
#     # Resize to target size for each sample in batch

#     resized_attn = F.interpolate(attention_map.unsqueeze(1), size=(192, 192), mode='bilinear', align_corners=False)

#     # Normalize to [0, 1] range for each sample
#     min_vals = resized_attn.amin(dim=[2, 3], keepdim=True)
#     max_vals = resized_attn.amax(dim=[2, 3], keepdim=True)
#     normalized_attn = (resized_attn - min_vals) / (max_vals - min_vals + 1e-8)
    
#     return normalized_attn

def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1,config=None,visdir=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    visual_freq = 500
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        GLA_img = samples['images']
        LLA_img = samples['aug_images']
        lbl = samples['labels']
        if cur_iteration % visual_freq == 0:
            visual_dict={}
            visual_dict['GLA']=GLA_img.detach().cpu().numpy()[0,0]
            visual_dict['LLA']=LLA_img.detach().cpu().numpy()[0,0]
            visual_dict['GT']=lbl.detach().cpu().numpy()[0]
        else:
            visual_dict=None

        lbl = lbl.to(device)
        input_var = Variable(GLA_img, requires_grad=True)
        input_var = preprocess_images(input_var, processor, device)
        # model.config.output_attentions = True

        optimizer.zero_grad()
        output = model(input_var, output_attentions=True)
        logits, attention_maps = output.logits, output.attentions
        # print(logits.shape, "logits shape")
        logits = F.interpolate(logits, size=lbl.shape[-2:], mode="bicubic", align_corners=False)
        # logits = output.logits
        # attention_maps = output.attentions
        # print(lbl.shape,"label shape")
        # lbl = lbl.unsqueeze(1)  # Add a channel dimension
        # lbl = F.interpolate(lbl.float(), size=((48, 48)), mode='nearest')
        # lbl = lbl.float()  # Convert to float if necessary
        # lbl = F.interpolate(lbl, size=((48, 48)), mode='nearest')
# Remove the channel dimension (squeeze) and convert to long
        # lbl = lbl.squeeze(1).long() 
        # print(lbl.shape,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&")
        loss_dict = criterion.get_loss(logits, lbl)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        losses.backward(retain_graph=True)
        # if isinstance(attention_maps, tuple):
        #   attention_maps = torch.stack(attention_maps, dim=0)  # Stack into a tensor
        #   attention_map = attention_maps.mean(dim=1)
        # print(attention_maps.shape)
        # attention_map = attention_maps.mean(dim=1)
        # Normalize the attention map to a [0, 1] range
        # gradient = torch.sqrt(torch.mean(attention_maps ** 2, dim=1, keepdim=True)).detach()
        # attention_map = attention_maps[-1]
        # print(attention_map.shape)
        # print(attention_maps[-2].shape)
        # print(GLA_img.shape)
        # gradient = torch.mean(attention_map.unsqueeze(1), dim=1)
        # print("gradient shape : ", gradient.shape)
          # Average across attention heads
        # attention_map = torch.nn.functional.interpolate(
        #     attention_map.unsqueeze(1), size=(GLA_img.shape[2], GLA_img.shape[3]), mode='bilinear', align_corners=False
        # ).squeeze()
        # saliency
        # gradient = torch.sqrt(torch.mean(attention_maps ** 2, dim=1, keepdim=True)).detach()
        # print("attention mapS :",attention_maps.shape)
        saliency=process_attention_maps(attention_maps)
        # saliency = get_SBF_map(gradient,config.grid_size)
        # print("saliency shape ",saliency.shape)
        if visual_dict is not None:
            visual_dict['GLA_pred']=torch.argmax(logits,1).cpu().numpy()[0]

        if visual_dict is not None:
            visual_dict['GLA_saliency']= saliency.detach().cpu().numpy()[0,0]

        
        mixed_img = GLA_img.detach() * saliency + LLA_img * (1 - saliency)

        if visual_dict is not None:
            visual_dict['SBF']= mixed_img.detach().cpu().numpy()[0,0]

        aug_var = Variable(mixed_img, requires_grad=True)
        aug_var = preprocess_images(aug_var,processor,device)
        aug_output  = model(aug_var)
        aug_logits = aug_output.logits
        aug_logits = F.interpolate(aug_logits, size=lbl.shape[-2:], mode="bicubic", align_corners=False)
        aug_loss_dict = criterion.get_loss(aug_logits, lbl)
        aug_losses = sum(aug_loss_dict[k] * criterion.weight_dict[k] for k in aug_loss_dict.keys() if k in criterion.weight_dict)

        aug_losses.backward()

        if visual_dict is not None:
            visual_dict['SBF_pred'] = torch.argmax(aug_logits, 1).cpu().numpy()[0]

        optimizer.step()

        all_loss_dict={}
        for k in loss_dict.keys():
            if k not in criterion.weight_dict:continue
            all_loss_dict[k]=loss_dict[k]
            all_loss_dict[k+'_aug']=aug_loss_dict[k]

        metric_logger.update(**all_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if cur_iteration>=max_iteration and max_iteration>0:
            break

        if visdir is not None and cur_iteration%visual_freq==0:
            fs=int(len(visual_dict)**0.5)+1
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs,fs,idx+1)
                plt.title(k)
                plt.axis('off')
                if k not in ['GT','GLA_pred','SBF_pred']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png')
            plt.close()
        cur_iteration+=1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    def convert_to_one_hot(tensor,num_c):
        return F.one_hot(tensor,num_c).permute((0,3,1,2))
    dices=[]
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        img = preprocess_images(img, processor, device)
        output = model(img)
        logits = output.logits
        logits = F.interpolate(logits, size=lbl.shape[-2:], mode="bicubic", align_corners=False)
        num_classes=logits.size(1)
        pred=torch.argmax(logits,dim=1)
        one_hot_pred=convert_to_one_hot(pred,num_classes)
        one_hot_gt=convert_to_one_hot(lbl,num_classes)
        dice=compute_meandice(one_hot_pred,one_hot_gt,include_background=False)
        dices.append(dice.cpu().numpy())
    dices=np.concatenate(dices,0)
    dices=np.nanmean(dices,0)
    return dices

def prediction_wrapper(model, test_loader, epoch, label_name, mode = 'base', save_prediction = False):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    device = torch.device('cuda')
    model.eval()
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        # recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['labels'].shape[0] == 1 # enforce a batchsize of 1

            img = batch['images'].cuda()
            gth = batch['labels'].cuda()

            img = preprocess_images(img, processor, device)
            output = model(img)
            pred = output.logits
            pred = F.interpolate(pred, size=gth.shape[-2:], mode="bicubic", align_corners=False)

            pred=torch.argmax(pred,1)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['images'][0, 0,...].numpy()
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name),label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper(vol_list, nclass, label_name):
    """
    Evaluatation and arrange predictions
    """
    def convert_to_one_hot2(tensor,num_c):
        return F.one_hot(tensor.long(),num_c).permute((3,0,1,2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices=compute_meandice(y_pred=convert_to_one_hot2(pred_,nclass),y=convert_to_one_hot2(gth_,nclass),include_background=True).cpu().numpy()[0].tolist()

        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    return error_dict, dsc_table, domain_names

