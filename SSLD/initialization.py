import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from SSLD import loss_function
from SSLD import train
from SSLD import utils
from SSLD.loss_function import csLoss

# One hot coding, we need this to initialize the probability model, for example, P(zt=i)=1, P(zt=j)=0, etc.   
def one_hot(hmm_z_all,num_tv):
    pp=np.zeros((hmm_z_all.shape[0],hmm_z_all.shape[1],num_tv))

    for kk in range(hmm_z_all.shape[0]):
        t_all=hmm_z_all[kk]
        for i in range(hmm_z_all.shape[1]):

            pp[kk,i,int(t_all[i])]=1
    return pp
# Initialization training
def run(model,rnninfer,behinfer,sharedecoder,optimizer,optimizer_rnn,optimizer_beh,optimizer_share, scheduler,scheduler_rnn,scheduler_beh,scheduler_share,X_train,y_train,beh_train,label_train,X_test,y_test,beh_test,label_test,beh_labels,beh_labels_test,num_tv,coef_cross_beh,epochs,batch_size,alp1,alp2,alp3,save_name,save_folder,save_fold,save_ckp,device):
    mse_all=np.ones(epochs)
    error_all=np.ones(epochs)
    mse_all_test=np.ones(epochs)
    error_all_test=np.ones(epochs)
    loss_save=np.ones(epochs)
    loss_neural_save=np.ones(epochs)
    loss_beh_save=np.ones(epochs)
    loss_private_save=np.ones(epochs)
    start_time=time.time()
    pos_test_save_all=np.zeros((epochs,y_test.shape[0],y_test.shape[1],num_tv))
    cs_criterion=csLoss(5)
    # cs_criterion = cs_criterion.to(device)
    pp=one_hot(label_train.cpu().detach().numpy()[:,:,0],num_tv)
    pp=torch.tensor(pp,dtype=label_train.dtype,device=device)

    tds = TensorDataset(X_train,y_train,beh_train,pp)
    data_loader = DataLoader(tds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    for epoch in range(epochs):
        model.train()
        rnninfer.train()
        behinfer.train()
        sharedecoder.train()
        loss_print=0
        loss_neural_print=0
        loss_beh_print=0
        loss_private_print=0
        for X_train_batch, y_train_batch, beh_train_batch,pp_batch in data_loader:
            optimizer.zero_grad()
            optimizer_rnn.zero_grad()
            optimizer_beh.zero_grad()
            optimizer_share.zero_grad()
            infer_dist,inferred_h,mean_out=rnninfer(y_train_batch)
            infer_dist_beh,inferred_beh,mean_out_beh=behinfer(beh_train_batch)
    
            global_latent,beh_private,neural_private,beh_final,neural_final=sharedecoder(inferred_h,inferred_beh)
            
            prob_ini,prob_all_s,prob_all_h,prob_all_y,gamma1,delta1,fwp,bwp,prob_all_beh = model(X_train_batch,y_train_batch,beh_train_batch,global_latent,beh_final,neural_final,device)
            t1,t2=loss_function.get_loss(gamma1,delta1,prob_ini,prob_all_s,prob_all_h,prob_all_y,prob_all_beh)
            posterior_lk=model.get_posterior_lk(fwp,bwp)
            
            
            cross_en_behavior=loss_function.get_cross_entropy(posterior_lk,pp_batch)
            loss_beh=cs_criterion(global_latent,beh_private)
            
            loss_neural=cs_criterion(global_latent,neural_private)
            loss_private=cs_criterion(beh_private,neural_private)
            prior = torch.distributions.Normal(0, 1e-4)
            log_lik_zero = prior.log_prob(beh_private).mean() + prior.log_prob(neural_private).mean()

            loss_all=-(t1.mean()+t2.mean()+coef_cross_beh*(cross_en_behavior)+infer_dist.entropy().mean()+infer_dist_beh.entropy().mean()+log_lik_zero)
            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(rnninfer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(behinfer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(sharedecoder.parameters(), 1.0)
            
            optimizer.step()
            optimizer_rnn.step()
            optimizer_beh.step()
            optimizer_share.step()
            
            loss_print+=loss_all.detach()

            loss_neural_print+=loss_neural.detach()
            loss_beh_print+=loss_beh.detach()
            loss_private_print+=loss_private.detach()
        if epoch%100==0:
            print(f"Epoch {epoch+1}/{epochs}, loss = {loss_all}")
            end_time = time.time()
            if epoch!=0:
                utils.compute_time(start_time,end_time,epochs,epoch+1)
        loss_save[epoch]=loss_print
        loss_neural_save[epoch]=loss_neural_print
        loss_beh_save[epoch]=loss_beh_print
        loss_private_save[epoch]=loss_private_print
        scheduler.step()
        scheduler_rnn.step()
        scheduler_beh.step()
        scheduler_share.step()
        y_pred_train,beh_pred_train,pos_train,_,_,_,_,_=train.eval_(model,rnninfer,behinfer,sharedecoder,X_train,y_train,beh_train,device)


        y_pred_test,beh_pred_test,pos_test,_,_,_,_,_=train.eval_(model,rnninfer,behinfer,sharedecoder,X_test,y_test,beh_test,device)

        pos_test_save_all[epoch]=pos_test


        if save_ckp==True:
            torch.save({
                'num_tv':num_tv,
    
                'alp1':alp1,
                'alp2':alp2,
                'alp3':alp3,
    
                'coef_cross_beh':coef_cross_beh,
    
                'y_test':y_test.cpu().detach().numpy(),
                'X_test':X_test.cpu().detach().numpy(),
                'beh_test':beh_test.cpu().detach().numpy(),
                'label_test':label_test.cpu().detach().numpy(),
    
                'model_state_dict': model.state_dict(),
                'rnninfer_state_dict': rnninfer.state_dict(),
                'behinfer_state_dict': behinfer.state_dict(),
                'sharedecoder_state_dict': sharedecoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mse_all':mse_all,
                'error_all':error_all,
                'mse_all_test':mse_all_test,
                'error_all_test':error_all_test,
                'loss_train':loss_save,
                'pos_test_all':pos_test_save_all,
                }, save_folder+'/ini_autosave_'+save_name+'_model_hidden'+str(model.hidden_shape)+'_'+str(int(alp1))+'_fold'+str(save_fold)+'.pt')
        
    return model,rnninfer,behinfer,sharedecoder,mse_all,error_all,mse_all_test,error_all_test,loss_save,pos_test_save_all