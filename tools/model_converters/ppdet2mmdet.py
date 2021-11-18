
import paddle
import torch
import numpy as np
import os

# picodet --> mmdet
bn_names_map = {
    '_mean': 'running_mean',
    '_variance': 'running_var',
}
neck_names_map = {
    'conv_t.convs': 'trans.trans',
    'dwconv': 'depthwise_conv.conv',
    'bn1': 'depthwise_conv.bn',
    'pwconv': 'pointwise_conv.conv',
    'bn2': 'pointwise_conv.bn',
}
## head

['0.0', '0.1', '1.0', '1.1', '2.0', '2.1', '3.0', '3.1']
head_names_map = {
    'distribution_project': 'integral',
    'head_cls': 'gfl_cls.',
    'conv_feat.cls_conv': 'cls_convs',
    'dw0.0.conv': '0.0.depthwise',
    'dw0.0.norm': '0.0.dwnorm',
    'pw0.0.conv': '0.0.pointwise',
    'pw0.0.norm': '0.0.pwnorm'
}




def paddle2torch(paddle_model, torch_model):
    paddle_param_state_dict = paddle.load(paddle_model)

    torch_param_state_dict = {}

    # bn names
    for k, v in paddle_param_state_dict.items():
        # print(k)
        if '_mean' in k:
            new_k = k.replace('_mean', 'running_mean')
        elif '_variance' in k:
            new_k = k.replace('_variance', 'running_var')
        else:
            new_k = k
        torch_param_state_dict[new_k] = torch.tensor(np.array(v))

    # neck
    old_ks = []
    new_ks = []
    for k, v in torch_param_state_dict.items():
        if 'neck' in k:
            if 'conv_t.convs' in k:
                new_k = k.replace('conv_t.convs', 'trans.trans')
            elif 'dwconv' in k:
                new_k = k.replace('dwconv', 'depthwise_conv.conv')
            elif 'bn1' in k:
                new_k = k.replace('bn1', 'depthwise_conv.bn')
            elif 'pwconv' in k:
                new_k = k.replace('pwconv', 'pointwise_conv.conv')
            elif 'bn2' in k:
                new_k = k.replace('bn2', 'pointwise_conv.bn')
            else:
                new_k = k

            

            # torch_param_state_dict[new_k] = v
            old_ks.append(k)
            new_ks.append(new_k)

    # head

    for k, v in torch_param_state_dict.items():

        if 'head' in k:
            splited_k = k.split('.')
            splited_k[0] = 'bbox_head'
            new_k = '.'.join(splited_k)

            if 'conv_feat.cls_conv_' in k:
                new_k = new_k.replace('conv_feat.cls_conv_', 'cls_convs.')
            if 'head_cls' in k:
                new_k = new_k.replace('head_cls', 'gfl_cls.')
            if 'distribution_project' in k:
                new_k = new_k.replace('distribution_project', 'integral')
            
            if 'dw' in k or 'pw' in k:
                for idx in ['0.0', '0.1', '1.0', '1.1', '2.0', '2.1', '3.0', '3.1']:

                    old_name =  'dw{}.conv'.format(idx)
                    if old_name in k:
                        new_name = '{}.depthwise'.format(idx)
                        break
                    old_name = 'dw{}.norm'.format(idx)
                    if old_name in k:
                        new_name = '{}.dwnorm'.format(idx)
                        break
                    old_name = 'pw{}.conv'.format(idx)
                    if old_name in k:
                        new_name = '{}.pointwise'.format(idx)
                        break
                    old_name = 'pw{}.norm'.format(idx)
                    if old_name in k:
                        new_name = '{}.pwnorm'.format(idx)
                        break
                new_k = new_k.replace(old_name, new_name)
            # torch_param_state_dict[new_k] = v
            old_ks.append(k)
            new_ks.append(new_k)
    for k, n_k, in zip(old_ks, new_ks):
        v  = torch_param_state_dict[k]
        torch_param_state_dict[n_k] = v
        if k != n_k:
            del torch_param_state_dict[k]

    torch.save(torch_param_state_dict, torch_model)


def det2cls(paddle_model, torch_model):
    paddle_param_state_dict = paddle.load(paddle_model)

    torch_param_state_dict = {}

    for k, v in paddle_param_state_dict.items():
        if k.startswith('backbone.'):
            k = k[9:]
            v = torch.tensor(np.array(v))
            torch_param_state_dict[k] = v

    torch.save(torch_param_state_dict, torch_model)



if __name__ == "__main__":

    # models = [
    #     'PPLCNet_x0_25_pretrained.pdparams',
    #     'PPLCNet_x0_35_pretrained.pdparams',
    #     'PPLCNet_x0_5_pretrained.pdparams',
    #     'PPLCNet_x0_75_pretrained.pdparams',
    #     'PPLCNet_x1_0_pretrained.pdparams',
    #     'PPLCNet_x1_5_pretrained.pdparams',
    #     'PPLCNet_x2_0_pretrained.pdparams',
    #     'PPLCNet_x2_5_pretrained.pdparams'
    # ]

    # root_dir = "/home/work/Data/pretrain/PP_LCNet"

    # for model in models:
    #     paddle_model = os.path.join(root_dir, model)
    #     torch_model = os.path.join(root_dir, model.replace('pdparams', 'pth'))

    #     paddle2torch(paddle_model, torch_model)


    models = [
        # "picodet_l_320_coco.pdparams",
        # "picodet_l_416_coco.pdparams",
        # "picodet_l_640_coco.pdparams",
        # "picodet_m_320_coco.pdparams",
        # "picodet_m_416_coco.pdparams",
        # "picodet_s_320_coco.pdparams",
        "picodet_s_416_coco.pdparams"
    ]

    # root_dir = "/mnt/Models/pretrained/ppdet"

    # for model in models:
    #     paddle_model = os.path.join(root_dir, model)
    #     # torch_model = os.path.join(root_dir, "esnet", model.replace('pdparams', 'pth').replace('picodet', 'esnet'))
    #     # det2cls(paddle_model, torch_model)

    #     torch_model = os.path.join(root_dir, model.replace('pdparams', 'pth'))
    #     paddle2torch(paddle_model, torch_model)



    torch_model = '/mnt/Models/pretrained/ppdet/picodet_s_416_coco.pth'

    model = torch.load(torch_model)

    for k, v in model.items():
        print(k)


