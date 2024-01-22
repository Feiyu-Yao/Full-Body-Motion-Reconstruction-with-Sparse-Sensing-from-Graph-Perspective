from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import numpy as np

from models.select_model import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss

from utils.utils_regularizers import regularizer_orth, regularizer_clip
from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from IPython import embed
from utils.utils_transform import bgs
from utils import utils_transform
from utils import utils_visualize as vis
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa


def sym_penalty(pred_out):
    """
    get penalty for the symmetry of human body
    :return:
    """
    loss_sym = 0
    left_bone = [(9,14),(14,17),(17,19),(19,21),(0,2),(2,5),(5,8),(8,11)]
    right_bone = [(9,13),(13,16),(16,18),(18,20),(0,1),(1,4),(4,7),(7,10)]
    for (i_left,j_left),(i_right,j_right) in zip(left_bone,right_bone):
        left_part = pred_out[:,i_left,:]-pred_out[:,j_left,:]
        right_part = pred_out[:, i_right, :] - pred_out[:, j_right, :]
        loss_sym += torch.mean(torch.norm(left_part, dim=- 1) - torch.norm(right_part, dim=- 1))
    
    return loss_sym


def square_norm(x):
    """
    Helper function returning square of the euclidean norm.
    Also here we clamp it since it really likes to die to zero.
    """
    norm = torch.norm(x, dim=-1, p=2) ** 2
    return torch.clamp(norm, min=1e-5)

def poincare_distance(x):
    '''
    Calculate pair-wise poincare distance between each row in two input tensors
    
    See equation (1) in this paper for mathematical expression:
    https://arxiv.org/abs/1705.08039
    '''
    (batch_shape, node, D) = x.shape
    a = (1 - square_norm(x)).view(batch_shape,node, 1)
    b = (1 - square_norm(x)).view(batch_shape,1, node)
    return torch.acosh(1 + 2 * pairwise_distances(x) / torch.matmul(a, b))



def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
            y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(x.shape[0], 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 2, 1)#.detach()
        y_norm = x_norm.view(x.shape[0], 1,x.shape[1])#.detach()
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)8
    return dist


def contrastive_loss(x, target):
    euc_dist = (pairwise_distances(x, target) ** 2)
    positive_dist = torch.diagonal(torch.exp(-euc_dist ),dim1=0,dim2=1)
    all_dist = torch.sum(torch.exp(- euc_dist),dim=1)
    loss = -1 * torch.log(positive_dist/all_dist)
    loss = loss.mean()
    

    return torch.clamp(loss, 1e-7, np.inf)


class ModelAvatar(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelAvatar, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.window_size = self.opt['netG']['window_size']
        self.bm = self.netG.module.body_model


        


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                           # load model
        self.log_dict = OrderedDict()         # log
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, test=False):
        load_path_G = self.opt['path']['pretrained_netG'] if test == False else self.opt['path']['pretrained']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        self.classify_loss = nn.CrossEntropyLoss()
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'geodesic':
            self.G_lossfn = geodesic_loss_R(reduction='mean')
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        number = 0
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
                number = number + v.nelement()
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)



    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True, test=False):
        self.L = data['L'].to(self.device)
        self.P = data['P']
        self.Head_trans_global = data['Head_trans_global'].to(self.device)
        self.H_global_orientation = data['H'].squeeze()[:,:6].to(self.device)
        self.H_joint_rotation = data['H'].squeeze()[:,6:].to(self.device)
#        self.H = torch.cat([self.H_global_orientation, self.H_joint_rotation],dim=-1).to(self.device)
        self.H_joint_position = self.netG.module.fk_module(self.H_global_orientation, self.H_joint_rotation , self.bm)
        if not test:
            self.hyper_close = data['ground_truth_for_distinct'].squeeze(1).cuda()

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # self.E_global_orientation, self.E_joint_rotation, self.E_joint_position, self.all_for_hyper_in_latent, self.target_in_hyper, self.hyperbolic_loss = self.netG(self.L, hyper_target = self.hyper_close )
        self.E_global_orientation, self.E_joint_rotation, self.E_joint_position, self.hyper_loss = self.netG(self.L, hyper_target = self.hyper_close )
    # def poincare_distance_tree(self, x):


    # ---------------------------------------- 
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        # self.classification_label = torch.range(0,21).unsqueeze(0).repeat(128,1).reshape(128*22,1).cuda()
        global_orientation_loss = self.G_lossfn(self.E_global_orientation, self.H_global_orientation)
        joint_rotation_loss = self.G_lossfn(self.E_joint_rotation, self.H_joint_rotation)
        joint_position_loss = self.G_lossfn(self.E_joint_position, self.H_joint_position)
        loss_sym = sym_penalty(self.E_joint_position)

        # contrastive_loss_add = contrastive_loss(self.all_for_hyper_in_latent, self.target_in_hyper)
        # loss = poincare_distance_tree(self.expended_latent)
        # classification_loss = self.classify_loss(self.classification.reshape(128*3,3), self.classification_label.long())
        loss =  0.02*global_orientation_loss + joint_rotation_loss + joint_position_loss + loss_sym # + self.hyper_loss #+ self.hyperbolic_loss#+ 0.5*classification_loss
        # loss = global_orientation_loss + joint_rotation_loss + joint_position_loss
        # loss=0.01*self.hyperbolic_loss
        loss.backward()


        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['total_loss'] = loss.item()
        self.log_dict['global_orientation_loss'] = global_orientation_loss.item()
        self.log_dict['joint_rotation_loss'] = joint_rotation_loss.item()
        self.log_dict['joint_position_loss'] = joint_position_loss.item()
        # self.log_dict['contrastive_loss_add'] = self.hyper_loss.item()
        # self.log_dict['hyperbolic_loss'] = self.hyperbolic_loss.item()
        
        # self.log_dict['classification_loss'] = classification_loss.item()


        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):

            self.netG.eval()

            self.L = self.L.squeeze()
      
            self.Head_trans_global = self.Head_trans_global.squeeze()
            window_size = self.opt['datasets']['train']['window_size']

            input_singleframe = False
            with torch.no_grad():

                if self.L.shape[0] < window_size:


                    if input_singleframe:
                        input_list = []
                        for frame_idx in range(0,self.L.shape[0]):
                            input_list.append(self.L[[frame_idx]].unsqueeze(0))
                        input_tensor = torch.cat(input_list, dim = 0)

                        E_global_orientation_tensor, E_joint_rotation_tensor = self.netG(input_tensor,do_fk=False)
                    else:
                        E_global_orientation_list = []  
                        E_joint_rotation_list = []                     

                        for frame_idx in range(0,self.L.shape[0]):
                            E_global_orientation, E_joint_rotation = self.netG(self.L[0:frame_idx+1].unsqueeze(0), do_fk=False)
                            E_global_orientation_list.append(E_global_orientation)
                            E_joint_rotation_list.append(E_joint_rotation)
                        E_global_orientation_tensor = torch.cat(E_global_orientation_list, dim=0)
                        E_joint_rotation_tensor = torch.cat(E_joint_rotation_list, dim=0)

                else:  


                    input_list_1 = []
                    input_list_2 = []

                    if input_singleframe:
                        for frame_idx in range(0,window_size):
                            input_list_1.append(self.L[[frame_idx]].unsqueeze(0))
                        input_tensor_1 = torch.cat(input_list_1, dim = 0)
                        E_global_orientation_1, E_joint_rotation_1 = self.netG(input_tensor_1,do_fk=False)
                    else:
                        E_global_orientation_list_1 = []  
                        E_joint_rotation_list_1 = []          
                        for frame_idx in range(0,window_size):
                            E_global_orientation, E_joint_rotation = self.netG(self.L[0:frame_idx+1].unsqueeze(0), do_fk=False)
                            E_global_orientation_list_1.append(E_global_orientation)
                            E_joint_rotation_list_1.append(E_joint_rotation)
                        E_global_orientation_1 = torch.cat(E_global_orientation_list_1, dim=0)
                        E_joint_rotation_1 = torch.cat(E_joint_rotation_list_1, dim=0)

                    for frame_idx in range(window_size,self.L.shape[0]):
                        input_list_2.append(self.L[frame_idx-window_size+1:frame_idx+1,...].unsqueeze(0))
                    input_tensor_2 = torch.cat(input_list_2, dim = 0)
                    E_global_orientation_2, E_joint_rotation_2 = self.netG(input_tensor_2,do_fk=False)

                    E_global_orientation_tensor = torch.cat([E_global_orientation_1,E_global_orientation_2], dim=0)
                    E_joint_rotation_tensor = torch.cat([E_joint_rotation_1,E_joint_rotation_2], dim=0)


            self.E_global_orientation = E_global_orientation_tensor
            self.E_joint_rotation = E_joint_rotation_tensor
            self.E = torch.cat([E_global_orientation_tensor, E_joint_rotation_tensor],dim=-1).to(self.device)

            predicted_angle = utils_transform.sixd2aa(self.E[:,:132].reshape(-1,6).detach()).reshape(self.E[:,:132].shape[0],-1).float()

            # Calculate global translation

            T_head2world = self.Head_trans_global.clone()
            T_head2root_pred = torch.eye(4).repeat(T_head2world.shape[0],1,1).cuda()
            rotation_local_matrot = aa2matrot(torch.cat([torch.zeros([predicted_angle.shape[0],3]).cuda(),predicted_angle[...,3:66]],dim=1).reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
            rotation_global_matrot = local2global_pose(rotation_local_matrot, self.bm.kintree_table[0][:22].long())
            head2root_rotation = rotation_global_matrot[:,15,:]

            body_pose_local_pred=self.bm(**{'pose_body':predicted_angle[...,3:66]})
            head2root_translation = body_pose_local_pred.Jtr[:,15,:]
            T_head2root_pred[:,:3,:3] = head2root_rotation
            T_head2root_pred[:,:3,3] = head2root_translation
            t_head2world = T_head2world[:,:3,3].clone()
            T_head2world[:,:3,3] = 0
            T_root2world_pred = torch.matmul(T_head2world, torch.inverse(T_head2root_pred))

            rotation_root2world_pred = matrot2aa(T_root2world_pred[:,:3,:3])
            translation_root2world_pred = T_root2world_pred[:,:3,3]
            body_pose_local=self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
            position_global_full_local = body_pose_local.Jtr[:,:22,:]
            t_head2root = position_global_full_local[:,15,:]
            well,_ = t_head2world.shape
            if well <= 3000:
                t_root2world = -t_head2root+t_head2world.cuda()

                self.predicted_body=self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world}) 
                # No stabilizer: 'root_orient':rotation_root2world_pred.cuda()

                self.predicted_position = self.predicted_body.Jtr[:,:22,:]
            
                self.predicted_angle = predicted_angle
                self.predicted_translation = t_root2world


            else:
                t_root2world = -t_head2root+t_head2world.cpu()

                self.predicted_body=self.bm(**{'pose_body':predicted_angle[...,3:66].cpu(), 'root_orient':predicted_angle[...,:3].cpu(), 'trans': t_root2world}) 
                # No stabilizer: 'root_orient':rotation_root2world_pred.cuda()

                self.predicted_position = self.predicted_body.Jtr[:,:22,:].cpu()
            
                self.predicted_angle = predicted_angle.cpu()
                self.predicted_translation = t_root2world.cpu()


            body_parms = self.P

            for k,v in body_parms.items():
                body_parms[k] = v.squeeze().cuda()
                body_parms[k] = body_parms[k][-self.E.shape[0]:,...]

            self.gt_body=self.bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','trans', 'root_orient']})
            self.gt_position = self.gt_body.Jtr[:,:22,:]

            self.gt_local_angle = body_parms['pose_body']
            self.gt_global_translation = body_parms['trans']
            self.gt_global_orientation = body_parms['root_orient']




            self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_prediction(self,):
        body_parms = OrderedDict()
        body_parms['pose_body'] = self.predicted_angle[...,3:66]
        body_parms['root_orient'] = self.predicted_angle[...,:3]
        body_parms['trans'] = self.predicted_translation
        body_parms['position'] = self.predicted_position
        body_parms['body'] = self.predicted_body

        return body_parms

    def current_gt(self, ):
        body_parms = OrderedDict()
        body_parms['pose_body'] = self.gt_local_angle
        body_parms['root_orient'] = self.gt_global_orientation
        body_parms['trans'] = self.gt_global_translation
        body_parms['position'] = self.gt_position
        body_parms['body'] = self.gt_body
        return body_parms


    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
