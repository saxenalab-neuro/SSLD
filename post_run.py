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
warnings.filterwarnings('ignore', message='enable_nested_tensor is True')


y_test_entire          = []
y_pred_test_entire     = []
beh_test_entire        = []
y_pred_test_beh_entire = []
test_data_label_entire = []
inferred_state_entire  = []
global_l_entire        = []
beh_l_entire           = []
neural_l_entire        = []

for jobid in range(5):
    m1 = torch.load(
        './result/area2_model_hidden8_0_50000_fold' + str(jobid) + '.pt',
        weights_only=False, map_location=torch.device('cpu')
    )

    test_data       = m1['y_test']
    beh_test_data   = m1['beh_test']
    test_data_label = m1['label_test']

    device = torch.device('cpu')
    dtype  = torch.float32

    y_test      = torch.tensor(test_data,     dtype=dtype, device=device)
    X_test      = torch.zeros(test_data.shape, dtype=dtype, device=device)
    beh_test    = torch.tensor(beh_test_data, dtype=dtype, device=device)
    label_test  = torch.tensor(test_data_label, dtype=dtype, device=device)

    input_shape          = X_test.shape[2]
    beh_shape            = beh_test.shape[2]
    num_tv               = m1['num_tv']
    hidden_shape         = m1['hidden_shape']
    bottleneck_shape     = m1['bottleneck_shape']
    beh_private_shape    = m1['beh_private_shape']
    neural_private_shape = m1['neural_private_shape']

    model        = model_srnn.Model(input_shape, beh_shape, num_tv, hidden_shape, beh_private_shape, neural_private_shape).to(device)
    rnninfer     = inference_network.RNNInfer(input_shape, hidden_shape).to(device)
    behinfer     = inference_network.BEHInfer(beh_shape, hidden_shape).to(device)
    sharedecoder = share.Decoder(hidden_shape, bottleneck_shape, hidden_shape, beh_private_shape, neural_private_shape)

    model.load_state_dict(m1['model_state_dict'])
    rnninfer.load_state_dict(m1['rnninfer_state_dict'])
    behinfer.load_state_dict(m1['behinfer_state_dict'])
    sharedecoder.load_state_dict(m1['sharedecoder_state_dict'])

    infer_dist,     inferred_h,   mean_out     = rnninfer(y_test)
    infer_dist_beh, inferred_beh, mean_out_beh = behinfer(beh_test)
    global_latent, beh_private, neural_private, beh_final, neural_final = sharedecoder(inferred_h, inferred_beh)

    global_l_entire.append(global_latent.cpu().detach().numpy())
    beh_l_entire.append(beh_private.cpu().detach().numpy())
    neural_l_entire.append(neural_private.cpu().detach().numpy())

    y_pred_test     = model.emission(neural_final).cpu().detach().numpy()
    y_pred_test_beh = model.behemission(beh_final).cpu().detach().numpy()

    prob_ini, prob_all_s, prob_all_h, prob_all_y, gamma1, delta1, fwp_test, bwp_test, prob_all_beh = \
        model(X_test, y_test, beh_test, global_latent, beh_final, neural_final, device)
    posterior_lk_test = model.get_posterior_lk(fwp_test, bwp_test)
    pos_test = posterior_lk_test.cpu().detach().numpy()

    y_test_entire.append(y_test.cpu().numpy())
    y_pred_test_entire.append(y_pred_test)
    beh_test_entire.append(beh_test.cpu().numpy())
    y_pred_test_beh_entire.append(y_pred_test_beh)
    test_data_label_entire.append(test_data_label)
    inferred_state_entire.append(np.argmax(pos_test, axis=-1))

y_test_entire          = np.concatenate(y_test_entire,          axis=0)
y_pred_test_entire     = np.concatenate(y_pred_test_entire,     axis=0)
beh_test_entire        = np.concatenate(beh_test_entire,        axis=0)
y_pred_test_beh_entire = np.concatenate(y_pred_test_beh_entire, axis=0)
test_data_label_entire = np.concatenate(test_data_label_entire, axis=0)
inferred_state_entire  = np.concatenate(inferred_state_entire,  axis=0)
global_l_entire        = np.concatenate(global_l_entire,        axis=0)
beh_l_entire           = np.concatenate(beh_l_entire,           axis=0)
neural_l_entire        = np.concatenate(neural_l_entire,        axis=0)

mse_neural = np.array([
    mean_squared_error(y_test_entire[i], y_pred_test_entire[i])
    for i in range(y_test_entire.shape[0])
])
mse_beh = np.array([
    mean_squared_error(beh_test_entire[i], y_pred_test_beh_entire[i])
    for i in range(y_pred_test_beh_entire.shape[0])
])

trial_neural = np.where(mse_neural == mse_neural.min())[0][0]
trial_beh    = np.where(mse_beh    == mse_beh.min())[0][0]

os.makedirs('./plot', exist_ok=True)

# ─── Neural reconstruction ───
fig = plt.figure(figsize=(3, 3))
ax  = plt.subplot(1, 1, 1)
y_true = y_test_entire[trial_neural].T
y_pred = y_pred_test_entire[trial_neural].T
gap    = np.full((5, y_true.shape[1]), np.nan)
combined = np.concatenate([y_true, gap, y_pred], axis=0)

cmap = plt.cm.turbo.copy()
cmap.set_bad(color='white')
vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

im = ax.imshow(combined, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xlim(-0.5, combined.shape[1] - 0.5)
ax.set_ylim(combined.shape[0] - 0.5, -0.5)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Neural Activity')
ax.set_xlabel('Time', fontsize=12)

n_true, n_pred, n_gap = y_true.shape[0], y_pred.shape[0], gap.shape[0]
ax.text(-20, n_true / 2,                    'True\nNeurons',      va='center', ha='center', fontsize=12, rotation=90)
ax.text(-20, n_true + n_gap + n_pred / 2,   'Predicted\nNeurons', va='center', ha='center', fontsize=12, rotation=90)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
cbar.set_ticklabels(['Low', 'Medium', 'High'], fontsize=12, rotation=90)
plt.savefig('./plot/neural_recon.pdf', dpi=512, bbox_inches='tight')
plt.show()

# ─── Behavior reconstruction ───
fig = plt.figure(figsize=(3, 3))
ax  = plt.subplot(1, 1, 1)
y_true = beh_test_entire[trial_beh].T
y_pred = y_pred_test_beh_entire[trial_beh].T
gap    = np.full((1, y_true.shape[1]), np.nan)
combined = np.concatenate([y_true, gap, y_pred], axis=0)

cmap = plt.cm.plasma.copy()
cmap.set_bad(color='white')
vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

im = ax.imshow(combined, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xlim(-0.5, combined.shape[1] - 0.5)
ax.set_ylim(combined.shape[0] - 0.5, -0.5)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Behavior')
ax.set_xlabel('Time', fontsize=12)

n_true, n_pred, n_gap = y_true.shape[0], y_pred.shape[0], gap.shape[0]
ax.text(-10, n_true / 2,                  'True',      va='center', ha='center', fontsize=12, rotation=90)
ax.text(-10, n_true + n_gap + n_pred / 2, 'Predicted', va='center', ha='center', fontsize=12, rotation=90)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
cbar.set_ticklabels(['Low', 'Medium', 'High'], fontsize=12, rotation=90)
plt.savefig('./plot/beh_recon.pdf', dpi=512, bbox_inches='tight')
plt.show()

# ─── State visualization ───
fig, axes = plt.subplots(2, 1, figsize=(4, 2))

ax = axes[0]
ax.imshow(test_data_label_entire, aspect='auto', cmap='coolwarm', interpolation='nearest')
ax.set_title('True States', fontsize=10)
ax.set_xlabel('')
ax.set_ylabel('Trial ID')
ax.set_xticks([])

ax = axes[1]
ax.imshow(inferred_state_entire, aspect='auto', cmap='coolwarm', interpolation='nearest')
ax.set_title('Inferred States', fontsize=10)
ax.set_xlabel('Time')
ax.set_ylabel('Trial ID')
ax.set_xticks([])

plt.tight_layout()
plt.savefig('./plot/states.pdf', dpi=512, bbox_inches='tight')
plt.show()
