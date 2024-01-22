import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import math
from utils import utils_transform
from models.SCINet import SCINet
from models.SCINet_body import SCINet_body
from models.interactive_nodes import interactive_nodes
from models.interactive_angle_and_position import interactive_angle_and_position
from models.interactive_angle_and_position_1 import interactive_angle_and_position_1
import numpy as np

from models.modulated_gcn_conv import ModulatedGraphConv
from models.graph_non_local import GraphNonLocal
from models.non_local_embedded_gaussian import NONLocalBlock2D

nn.Module.dump_patches = True




class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv =  ModulatedGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


def square_norm(x):
    """
    Helper function returning square of the euclidean norm.
    Also here we clamp it since it really likes to die to zero.
    """
    norm = torch.norm(x, dim=-1, p=2) ** 2

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

list_left_up = [14,17,19,21,23]
list_right_up = [13,16,18,20,22]
list_left_down = [2,5,8,11]
list_right_down = [3,4,7,10]
middle = [0,6,9,15]

tree_hierarchical = [
    [0,[3,6,9,12,15],[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]],
    [1,[4,7,10],[0,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]],
    [2,[5,8,11],[0,1,3,4,6,7,9,10,12,13,14,15,16,17,18,19,20,21]],
    [3,[0,6,9,12,15],[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]],
    [4,[1,7,10],[0,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]],
    [5,[2,8,11],[0,1,3,4,6,7,9,10,12,13,14,15,16,17,18,19,20,21]],
    [6,[0,3,9,12,15],[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]],
    [7,[1,4,10],[0,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]],
    [8,[2,5,11],[0,1,3,4,6,7,9,10,12,13,14,15,16,17,18,19,20,21]],
    [9,[0,3,6,12,15],[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]],
    [10,[1,4,7],[0,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]],
    [11,[2,5,8],[0,1,3,4,6,7,9,10,12,13,14,15,16,17,18,19,20,21]],
    [12,[0,3,6,9,15],[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]],
    [13,[16,18,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,19,21]],
    [14,[17,19,21],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,20]],
    [15,[0,3,6,9,12],[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]],
    [16,[13,18,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,19,21]],
    [17,[14,19,21],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,20]],
    [18,[13,16,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,19,21]],
    [19,[14,17,21],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,20]],
    [20,[13,16,18],[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,19,21]],
    [21,[14,17,19],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,20]],
    ]

tree_positive_mask = torch.zeros([22,22]).cuda()
tree_negative_mask = torch.zeros([22,22]).cuda()
for i, j in enumerate(tree_hierarchical):
    tree_positive_mask[j[0],j[1]] = 1
    tree_negative_mask[j[0],j[2]] = 1


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
            y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(2).view(x.shape[0], x.shape[1], 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 2, 1)
        y_norm = x_norm.view(x.shape[0], 1,x.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist


def contrastive_loss(x):
    euc_dist = (pairwise_distances(x) ** 2)
    positive_dist = torch.diagonal(torch.exp(- torch.matmul(euc_dist , tree_positive_mask.T) ),dim1=1,dim2=2)
    all_dist = torch.diagonal(torch.exp(- torch.matmul(euc_dist , tree_negative_mask.T )),dim1=1,dim2=2) + positive_dist
    loss = -1 * torch.log(positive_dist/all_dist)
    loss = loss.mean()
    

    return torch.clamp(loss, 1e-7, np.inf)


class Avatar(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, body_model, device, adj):
        super(Avatar, self).__init__()
        self.linear_angle = nn.Linear(12,15)
        self.linear_pos = nn.Linear(6,15)
        coords_dim = (192,6)
        nodes_group = None
        hid_dim = 192
        hid_dim_2 = 60
        p_dropout = 0.2
        num_layers = 1
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_output = ModulatedGraphConv(hid_dim, hid_dim_2, adj) 
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = ModulatedGraphConv(hid_dim, coords_dim[1], adj) 
        self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)

        self.model_one = SCINet(
                output_len=16,
                input_len=40,
                input_dim= 144,
                hid_size = 4,
                num_stacks=2,
                num_levels=2,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = False,
                modified = True,
                RIN=False)

        self.model_two = SCINet(
                output_len=4,
                input_len=16,
                input_dim= 288,
                hid_size = 4,
                num_stacks=2,
                num_levels=2,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = False,
                modified = True,
                RIN=False)


        self.model_three = SCINet(
                output_len=16,
                input_len=40,
                input_dim= 144,
                hid_size = 4,
                num_stacks=2,
                num_levels=2,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = False,
                modified = True,
                RIN=False)

        self.model_four = SCINet(
                output_len=4,
                input_len=16,
                input_dim= 288,
                hid_size = 4,
                num_stacks=2,
                num_levels=2,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = False,
                modified = True,
                RIN=False)



        self.model_interactive_angle_and_position = interactive_angle_and_position(
                output_len=0,
                input_len=0,
                input_dim= 123,
                hid_size = 4,
                num_stacks=1,
                num_levels=5,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = False,
                modified = True,
                RIN=False)

        self.model_interactive_angle_and_position_1 = interactive_angle_and_position_1(
                output_len=0,
                input_len=0,
                input_dim= 192,
                hid_size = 4,
                num_stacks=1,
                num_levels=5,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = False,
                modified = True,
                RIN=False)



        self.list_left_up = [0,3,6,9,14,17,19,21]
        self.list_right_up = [0,3,6,9,13,16,18,20]
        self.list_left_down = [0,2,5,8,11]
        self.list_right_down = [0,1,4,7,10]
        self.middle = [0,3,6,9,12,15]

        self.body_model = body_model



    @staticmethod
    def fk_module(global_orientation, joint_rotation, body_model):

        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1,6)).reshape(global_orientation.shape[0],-1).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1,6)).reshape(joint_rotation.shape[0],-1).float()
        body_pose = body_model(**{'pose_body':joint_rotation, 'root_orient':global_orientation})
        joint_position = body_pose.Jtr

        return joint_position


    @staticmethod
    def ik_module(smpl, smpl_jids, target_pose_ids, target_3ds,
                         body_pose = None, global_orient = None, transl = None, learning_rate=1e-1, n_iter=5):
        target_3ds = target_3ds.view(1, -1, 3)
        body_pose_sub = torch.tensor(body_pose[:, target_pose_ids],requires_grad = True)
        opti_param = [body_pose_sub]
        optimiser = torch.optim.Adam(opti_param, lr = learning_rate)

        for i in range(n_iter):
            body_pose[:, target_pose_ids] = body_pose_sub
            out = smpl(**{'pose_body':body_pose, 'root_orient':global_orient, 'trans': transl})
            j_3ds = out.Jtr.view(1, -1, 3)
            loss = torch.mean(torch.sqrt(torch.sum(torch.square(j_3ds[:, smpl_jids].squeeze()-target_3ds)[:,[20,21],:],axis=-1)))

            optimiser.zero_grad()
        return body_pose


    def forward(self, x_before, do_fk = True ,hyper_target =None):

        self.batch_size, time_length,nodes,latent = x_before.shape
        if time_length == 1:
            x_before = torch.concat([x_before[:,0,:,:].repeat(1, 41-time_length,1 ,1),x_before],dim = 1) 
        elif time_length != 41:
            x_before = torch.concat([x_before[:,0:1,:,:].repeat(1, 41-time_length,1 , 1),x_before],dim = 1)
        x_angle = x_before[:,:,:,0:12].repeat(1,1,1,2)
        x_pos = x_before[:,:,:,12:18].repeat(1,1,1,4)
        x = torch.concat([x_angle,x_pos],dim = -1).permute(0,3,1,2).reshape(x_angle.shape[0] , 48,x_angle.shape[1] * 3)
        x = self.model_interactive_angle_and_position(x).reshape(x.shape[0],48,41,3).permute(0,2,3,1)[:,1:,:,:].reshape([x.shape[0], 40, 144]).cuda()
        x_list_1 = self.model_one(x)
        x_list_middle_1 = torch.zeros([x_list_1[0].shape[0],x_list_1[0].shape[1],x_list_1[0].shape[2]*2]).cuda()
        x_list_middle_1[:,:,0:287:2] = x_list_1[0]
        x_list_middle_1[:,:,1:288:2] = x_list_1[1]
        x_list_1 = self.model_two(x_list_middle_1)
        x_list_middle_1 = torch.zeros([x_list_1[0].shape[0],x_list_1[0].shape[1],x_list_1[0].shape[2]*2]).cuda()
        x_list_middle_1[:,:,0:575:2] = x_list_1[0]
        x_list_middle_1[:,:,1:576:2] = x_list_1[1]
        x_list_1 = x_list_middle_1.reshape(x_list_middle_1.shape[0], 4, 12, 48).permute(0,2,1,3).reshape(x.shape[0], 12, 192)[:,:11,:]
        x_list_2 = self.model_three(x)
        x_list_middle_2 = torch.zeros([x_list_2[0].shape[0],x_list_2[0].shape[1],x_list_2[0].shape[2]*2]).cuda()
        x_list_middle_2[:,:,0:287:2] = x_list_2[0]
        x_list_middle_2[:,:,1:288:2] = x_list_2[1]
        x_list_2 = self.model_four(x_list_middle_2)
        x_list_middle_2 = torch.zeros([x_list_2[0].shape[0],x_list_2[0].shape[1],x_list_2[0].shape[2]*2]).cuda()
        x_list_middle_2[:,:,0:575:2] = x_list_2[0]
        x_list_middle_2[:,:,1:576:2] = x_list_2[1]
        x_list_2 = x_list_middle_2.reshape(x_list_middle_2.shape[0], 4, 12, 48).permute(0,2,1,3).reshape(x.shape[0], 12, 192)[:,:11,:]
        x_list_3 = torch.concat([x_list_1.detach(), x_list_2],dim=1).cuda()
        x_list_3 = self.model_interactive_angle_and_position_1(x_list_3).cuda()



        all_joint_new = torch.zeros(([x_list_2.shape[0], x_list_2.shape[1] * 2, x_list_2.shape[2]])).cuda()
        all_joint_new[:,[0,1,2,3,4,5,6,9,12,13,14],:] = x_list_1
        all_joint_new[:,[7, 8, 10, 11, 15, 16, 17, 18, 19, 20, 21],:] = x_list_3
        loss = 0 


        out = self.gconv_layers(all_joint_new)
        out = out.unsqueeze(2)
        out = out.permute(0,3,2,1)
        out = self.non_local(out)
        out = out.permute(0,3,1,2)
        out = out.squeeze(3)
        out = self.gconv_output(out)


        global_orientation = out[:,0,:6].squeeze(1)
        joint_rotation = out[:,1:].reshape(self.batch_size,-1).contiguous()


        if do_fk:
            joint_position = self.fk_module(global_orientation, joint_rotation, self.body_model)

            return global_orientation, joint_rotation, joint_position, loss
        else:
            return global_orientation, joint_rotation


