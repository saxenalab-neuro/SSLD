import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from SSLD import model_srnn
from SSLD import inference_network
from SSLD import loss_function
from SSLD import share
from sklearn.metrics import mean_squared_error
import warnings
import sklearn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore', message='enable_nested_tensor is True')



def minmax_scale(arr, axis=None, ignore_nan=True):
    arr = np.asarray(arr)

    if ignore_nan:
        min_val = np.nanmin(arr, axis=axis, keepdims=True)
        max_val = np.nanmax(arr, axis=axis, keepdims=True)
    else:
        min_val = np.min(arr, axis=axis, keepdims=True)
        max_val = np.max(arr, axis=axis, keepdims=True)

    denom = max_val - min_val

    # avoid division by zero (constant values)
    denom = np.where(denom == 0, 1, denom)

    scaled = (arr - min_val) / denom
    return scaled


y_test_entire          = []
y_pred_test_entire     = []
beh_test_entire        = []
y_pred_test_beh_entire = []
test_data_label_entire = []
inferred_state_entire  = []
global_l_entire        = []
beh_l_entire           = []
neural_l_entire        = []

for jobid in range(1):
    m1=torch.load('./result/area2_entire_model_hidden8_0_50000_fold0.pt',weights_only=False,map_location=torch.device('cpu'))
    
    test_data=m1['y_test']
    beh_test_data=m1['beh_test']
    test_data_label=m1['label_test']
    #########################################################################################
    device = torch.device('cpu')
    dtype = torch.float32
    y_test=torch.tensor(test_data,dtype=dtype,device=device)
    X_test=torch.tensor(0*test_data,dtype=dtype,device=device)
    beh_test=torch.tensor(beh_test_data,dtype=dtype,device=device)
    label_test=torch.tensor(test_data_label,dtype=dtype,device=device)
    
    input_shape=X_test.shape[2] # Input shape of SRNNs, but the models are input free.
    beh_shape=beh_test.shape[2]
    num_tv=m1['num_tv'] # Number of RNNs in SRNNs.
    hidden_shape=m1['hidden_shape'] # Number of hidden states of SRNNs.
    bottleneck_shape=m1['bottleneck_shape']
    beh_private_shape=m1['beh_private_shape']
    neural_private_shape=m1['neural_private_shape']
    
    model = model_srnn.Model(input_shape,beh_shape,num_tv,hidden_shape,beh_private_shape,neural_private_shape).to(device)
    rnninfer=inference_network.RNNInfer(input_shape,hidden_shape).to(device)
    behinfer=inference_network.BEHInfer(beh_shape,hidden_shape).to(device)
    sharedecoder=share.Decoder(hidden_shape,bottleneck_shape,hidden_shape,beh_private_shape,neural_private_shape)
    
    model.load_state_dict(m1['model_state_dict'])
    rnninfer.load_state_dict(m1['rnninfer_state_dict'])
    behinfer.load_state_dict(m1['behinfer_state_dict'])
    sharedecoder.load_state_dict(m1['sharedecoder_state_dict'])
    
    infer_dist,inferred_h,mean_out=rnninfer(y_test)
    infer_dist_beh,inferred_beh,mean_out_beh=behinfer(beh_test)
    global_latent,beh_private,neural_private,beh_final,neural_final=sharedecoder(inferred_h,inferred_beh)
    global_l_entire.append(global_latent.cpu().detach().numpy())
    beh_l_entire.append(beh_private.cpu().detach().numpy())
    neural_l_entire.append(neural_private.cpu().detach().numpy())
    y_pred_test=model.emission(neural_final[:,:,:]).cpu().detach().numpy()
    y_pred_test_beh=model.behemission(beh_final[:,:,:]).cpu().detach().numpy()

    prob_ini,prob_all_s,prob_all_h,prob_all_y,gamma1,delta1,fwp_test,bwp_test,prob_all_beh = model(X_test,y_test,beh_test,global_latent,beh_final,neural_final,device)
    t1,t2=loss_function.get_loss(gamma1,delta1,prob_ini.cpu(),prob_all_s,prob_all_h,prob_all_y,prob_all_beh)
    posterior_lk_test=model.get_posterior_lk(fwp_test,bwp_test)
    pos_test=posterior_lk_test.cpu().detach().numpy()
    # 
    y_test_entire.append(y_test.cpu().numpy())
    y_pred_test_entire.append(y_pred_test)
    beh_test_entire.append(beh_test.cpu().numpy())
    y_pred_test_beh_entire.append(y_pred_test_beh)
    test_data_label_entire.append(test_data_label)
    inferred_state_entire.append(np.argmax(pos_test[:],axis=-1))

y_test_entire          = np.concatenate(y_test_entire,          axis=0)
y_pred_test_entire     = np.concatenate(y_pred_test_entire,     axis=0)
beh_test_entire        = np.concatenate(beh_test_entire,        axis=0)
y_pred_test_beh_entire = np.concatenate(y_pred_test_beh_entire, axis=0)
test_data_label_entire = np.concatenate(test_data_label_entire, axis=0)
inferred_state_entire  = np.concatenate(inferred_state_entire,  axis=0)
global_l_entire        = np.concatenate(global_l_entire,        axis=0)
beh_l_entire           = np.concatenate(beh_l_entire,           axis=0)
neural_l_entire        = np.concatenate(neural_l_entire,        axis=0)


ori_neural=np.load("./data/act_save_single_200.npy",allow_pickle=True)
ori_neural_all=[]
for k in range(len(ori_neural)):
    ori_neural_all.append(ori_neural[k])
ori_neural_all=np.concatenate(ori_neural_all,axis=0)
ori_neural_all=ori_neural_all/(np.abs(ori_neural_all).max())



ori_flat = ori_neural_all.reshape(ori_neural_all.shape[0], -1)
y_flat   = y_test_entire.reshape(y_test_entire.shape[0], -1)

matched_ori_idx = []
matched_y_idx = []
used_ori = set()

for y_idx, row in enumerate(y_flat):
    matches = np.where(np.all(np.isclose(ori_flat, row, atol=1e-6, equal_nan=True), axis=1))[0]
    matches = [m for m in matches if m not in used_ori]

    if len(matches) == 0:
        print(f'No match found for y row {y_idx}')
        continue

    ori_idx = matches[0]
    matched_ori_idx.append(ori_idx)
    matched_y_idx.append(y_idx)
    used_ori.add(ori_idx)

matched_ori_idx = np.array(matched_ori_idx)
matched_y_idx = np.array(matched_y_idx)

# sort by original order in ori_neural_all
sort_order = np.argsort(matched_ori_idx)

beh_test_entire_reorder = beh_test_entire[matched_y_idx][sort_order]
y_test_entire_reorder   = y_test_entire[matched_y_idx][sort_order]
global_l_entire_reorder = global_l_entire[matched_y_idx][sort_order]
ori_positions   = matched_ori_idx[sort_order]



distance=np.zeros((193,193))
for i in range(193):
    base_=beh_test_entire_reorder[i,:]
    for j in range(193):
        distance[i,j]=sklearn.metrics.mean_squared_error(beh_test_entire_reorder[j,:],base_)

beh_sim_all=-distance
neural_sim_all=np.corrcoef(global_l_entire_reorder.reshape(global_l_entire_reorder.shape[0],-1))



n_trials = neural_sim_all.shape[0]


upper_idx = np.triu_indices(n_trials, k=1)
r_s, p = pearsonr(minmax_scale(neural_sim_all[upper_idx]), minmax_scale(beh_sim_all[upper_idx]))
print(f'RSA correlation: r={r_s:.4f}, p={p:.2e}')

# plot
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
# vmin, vmax = 0.8, 1
im0 = axes[0].imshow(neural_sim_all, aspect='auto' ,cmap='magma_r')
axes[0].set_title('Shared Latent')
axes[0].set_xlabel('Trial')
axes[0].set_ylabel('Trial')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(beh_sim_all, aspect='auto',cmap='magma_r')
axes[1].set_title('Condition')
axes[1].set_xlabel('Trial')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# scatter of upper triangle values
axes[2].scatter(neural_sim_all[upper_idx][~np.isnan(neural_sim_all[upper_idx])], beh_sim_all[upper_idx][~np.isnan(beh_sim_all[upper_idx])], s=1, alpha=0.3)
axes[2].set_xlabel('Neural similarity')
axes[2].set_ylabel('Condition similarity')
axes[2].set_title(f'r={r_s:.3f}')

plt.tight_layout()
plt.savefig('./plot/rsa_comparison.pdf', dpi=512, bbox_inches='tight')
plt.show()


y_test_entire_reorder = y_test_entire[matched_y_idx][sort_order]
neural_sim_all=np.corrcoef(y_test_entire_reorder.reshape(y_test_entire_reorder.shape[0],-1))
n_trials = neural_sim_all.shape[0]
upper_idx = np.triu_indices(n_trials, k=1)
r_n, p = pearsonr(minmax_scale(neural_sim_all[upper_idx]), minmax_scale(beh_sim_all[upper_idx]))


beh_l_entire_reorder = beh_l_entire[matched_y_idx][sort_order]
neural_sim_all=np.corrcoef(beh_l_entire_reorder.reshape(beh_l_entire_reorder.shape[0],-1))
n_trials = neural_sim_all.shape[0]
upper_idx = np.triu_indices(n_trials, k=1)
r_b, p = pearsonr(minmax_scale(neural_sim_all[upper_idx]), minmax_scale(beh_sim_all[upper_idx]))


neural_l_entire_reorder = neural_l_entire[matched_y_idx][sort_order]
neural_sim_all=np.corrcoef(neural_l_entire_reorder.reshape(neural_l_entire_reorder.shape[0],-1))
n_trials = neural_sim_all.shape[0]
upper_idx = np.triu_indices(n_trials, k=1)
r_np, p = pearsonr(minmax_scale(neural_sim_all[upper_idx]), minmax_scale(beh_sim_all[upper_idx]))


r_plot=np.array([r_s,r_np,r_b,r_n])
color_list=['#ebf5e6','#eaf6fc','#fff3e0','#4d73b1','#4d73b1']
values = r_plot
labels = ['Shared', 'PN','PB','Neural']

fig, ax = plt.subplots(figsize=(3.5,3))

bars = ax.bar(np.arange(len(values)), values,color=color_list, width=0.5)

for i, v in enumerate(values):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)

ax.set_xticks(np.arange(len(values)))
ax.set_xticklabels(labels, rotation=25,fontsize=12, ha='right')
ax.set_ylabel('Correlation coefficient',fontsize=14)
ax.set_title('Similarity Comparison',fontsize=16)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.5, 1.0])
ax.tick_params(axis='y', labelsize=12)

# remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('./plot/similarity_compare.pdf',dpi=512, bbox_inches='tight')
plt.tight_layout()
plt.show()




color_list=['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
n_trials, n_time, n_feat = global_l_entire.shape
reshaped = global_l_entire.reshape(-1, n_feat)

# PCA to 3 components
pca = PCA(n_components=3)
reduced = pca.fit_transform(reshaped)

# reshape back to (27, 236, 3)
global_l_entire_pca = reduced.reshape(n_trials, n_time, 3)


ax = plt.figure(figsize=(3,3)).add_subplot(projection='3d')

# track whether we've already added a label for each color
labeled = [False, False,False,False,False]
label_names = ['Pre Move', 'Move']
for trial_neural in range(len(inferred_state_entire)):
    
    for i in range(len(y_pred_test[0])-2):
        state = int(inferred_state_entire[trial_neural][i])
        label = label_names[state] if not labeled[state] else None
        
        ax.plot(global_l_entire_pca[trial_neural,i:i+2,0],
                global_l_entire_pca[trial_neural,i:i+2,1],
                global_l_entire_pca[trial_neural,i:i+2,2],
                lw=0.5, color=color_list[state], label=label)
        
        if label:
            labeled[state] = True

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# ax.legend(fontsize=8, loc='best')
# ax.view_init(elev=20, azim=120) 
ax.view_init(elev=90, azim=120) 
plt.savefig('./plot/share_visual.pdf',dpi=512, bbox_inches='tight')
plt.show()