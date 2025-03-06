import torch
import torch.nn.functional as F
import numpy as np

def IBS_gpu(bss, model, target_class_0, target_class_1, data, precision = 0.05, device = 'cuda', noise_mul = 2, max_iters = None, shuffle = False):
    '''
    Informed Baseline Search with a starting point, and a pool of targets computes the inner boundary (baselines) of a NN

    :param bss: starting point, can be a batch
    :param model: model to be investigated
    :param target_class_0/target_class_1: data idx of the target pool for class 0 and class 1
    :param data: input samples of the network, used to navigare in the feature space
    :param precision: precision of the baseline (.5-precision < predict(bs) < .5+precision needs to be met to consider baseline)
    :param device: device
    :param noise_mul: scalar factor to modify the step intensity
    :param max_iters: max number of iteration until break
    :param shuffle: True for shuffle the target pool order


    :returns baselines: baselines computed
    '''
    #set up
    bss.to(device)
    model.to(device)
    model.eval()
    shape_offset = (len(bss.shape[1:]))
    low_b = np.min([len(target_class_0), len(target_class_1)])
    pred_bound = 1/2 # assuming two classes 

    ONE_NODE_OUT = model[-1].weight.shape[0] == 1 # extracting output dims 
    get_class = lambda x: torch.round(x, decimals=0) if ONE_NODE_OUT else lambda x: torch.argmax(x, dim=1)
    act_func = F.sigmoid if ONE_NODE_OUT else lambda y: F.softmax(y, dim=1)

    #shuffle targets
    if shuffle:
        np.random.shuffle(target_class_0)
        np.random.shuffle(target_class_1)
    #setup target pools
    target_classes = torch.stack([torch.tensor(target_class_0[:low_b]),torch.tensor(target_class_1[:low_b])], dim=1).to(device)
    #pool idx counter
    iters_count = torch.zeros((bss.shape[0],2),dtype=torch.int64, device=device)
    baselines = []
    iteration = 0

    while(bss.shape[0]>0 and (iteration < max_iters if not max_iters is None else True)):
        #computing prediction
        outs = model(bss)
        softmax_preds = act_func(outs)
        preds = softmax_preds[:,0]
        #Baseline check and extraction
        cond = ((pred_bound-precision) < preds)  * (preds <(pred_bound+precision))
        baselines.extend([*[x.cpu() for x in bss[torch.where(cond)]]])
        bss = bss[torch.where(~cond)]
        # jump out if no more data 
        if all(cond):break
        #getting rid of baseline information
        iters_count = iters_count[torch.where(~cond)]
        #computing next step
        idxs = get_class(softmax_preds[torch.where(~ cond)]).type(torch.bool)
        new_idx = torch.zeros(bss.shape[0], dtype=torch.int32,device=device) 
        new_idx = target_classes[iters_count[torch.arange(idxs.shape[0]), (~ idxs).long().squeeze()] %low_b,(~ idxs).long().squeeze()].to(device)
        #update idx counter
        iters_count[torch.arange(idxs.shape[0]), (~ idxs).long().squeeze()] +=1
        #compute directions
        dirs = torch.tensor(data[new_idx.cpu()].clone().detach(), device=device, dtype=torch.float32).view(bss.shape)
        dirs =dirs- bss 
        #compute intensity
        multi = ((max_iters/2-iteration) / max_iters/2)*noise_mul
        pred_values =  softmax_preds[torch.where(~ cond)[0]].squeeze()
        intensity = torch.round(((pred_values-(1-pred_values))/2) * multi, decimals=5)
        intensity = abs(intensity)
        #step
        weighted_input = dirs*intensity.view(-1, *[1]*shape_offset)
        bss = bss + weighted_input
        iteration += 1
    
    print(bss.shape[0], iteration )
    del weighted_input, dirs, bss
    return torch.stack(baselines) if len(baselines)>0 else []


def scaling_function(x): 
    return (2000/2-x) / 2000/2


def IBS(bs, model, target_class_0, target_class_1, precision = 0.05, iter = False, ret_fail = False, device = 'cpu', noise_mul = 2, path:bool = True, sc_func = scaling_function):
    '''
    Informed Baseline Search with a starting point, and a pool of targets computes a sample of the inner boundary (baseline) of a NN
        
    :param bs: starting point
    :param model: model to be investigated
    :param target_class_0/target_class_1: data of the target pool for class 0 and class 1, can be a single sample for each class
    :param precision: precision of the baseline (.5-precision < predict(bs) < .5+precision needs to be met to consider baseline)
    :param iter: True if target_class_0 and target_class_0 are iterators, otherwise false
    :param ret_fail: True to return 1 if the IBS failed to find a DB sample
    :param device: device
    :param noise_mul: scalar factor to modify the step intensity
    :param path: True to return all pointed computed in the search
    :param sc_func: Scaling function to apply to the step, default `(2000/2-x) / 2000/2`

    
    :returns bs: baseline computed
    :returns temp_bs: path computed if path is false returns the starting point
    :returns fail: (optional) 1 if IBS fails otherwhise 0
    
    '''
    
	# setup
    pred_bound = 1/2 # assuming two classes 
    ONE_NODE_OUT = model[-1].weight.shape[0] == 1 # extracting output dims 
    act_func = F.sigmoid if ONE_NODE_OUT else lambda y: F.softmax(y, dim=1)
    model = model.to(device)
    if not isinstance(bs, torch.Tensor): bs = torch.tensor(bs, dtype=torch.float32)
    bs = bs.to(device)
    acc=[]
    temp_bs = [bs.cpu().numpy()]
    out = model(bs)
    pred = act_func(out)[0].cpu().detach().numpy()

	# search phase
    while(True):
        acc.append( pred )
        # check bs predict
        if ((pred_bound-precision) < pred ).all() and (pred <(pred_bound+precision)).all():
            if path: temp_bs.append(bs.cpu().numpy())

            return (bs, temp_bs,0) if ret_fail else (bs, temp_bs)
            
		# class target selection 
        softmax_pred = act_func(out)
        pred_idx = softmax_pred.round() if ONE_NODE_OUT else torch.argmax(softmax_pred, dim=1)
        pred_value = softmax_pred if ONE_NODE_OUT else softmax_pred[0,pred_idx]
        
		# target extraction
        try:
            if iter:
                i1_idx = next(target_class_0) if pred_idx else next(target_class_1)
                i1 = [i1_idx]
            else: 
                i1 = target_class_0 if pred_idx else target_class_1
            
        except StopIteration: print("out of data"); break

        i1 = torch.tensor(i1, dtype=torch.float32).to(device)
        # compute step
        direction = (i1-bs)
        multi = sc_func(len(acc))*noise_mul
        intensity = round(((pred_value-(1-pred_value))/2).item()*multi,5)
        intensity = abs(intensity) #if ONE_NODE_OUT else abs(intensity)
        weighted_input = direction*intensity
        bs = bs + weighted_input
        
        if path:temp_bs.append(bs.cpu().numpy())
        out = model(bs)
        pred = act_func(out)[0].cpu().detach().numpy()

    return (bs,temp_bs,1) if ret_fail else (bs,temp_bs)
