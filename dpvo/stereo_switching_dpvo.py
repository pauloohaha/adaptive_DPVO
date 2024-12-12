import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3, SO3

from .net import VONet
from .utils import *
from . import projective_ops as pops

import math

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        self.left_right = [0]*self.N #0 for left, 1 for right

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None
        if viz:
            self.start_viewer()

        # dynamic debug
        self.dynamic_debug_delta_average_log = [] #average delta of the current sampled frame
        self.dynamic_debug_depth_average_log = [] #average depth of the current sampled frame
        self.dynamic_debug_confidence_average_log = [] #average confidence of the current sampled frame
        self.dynamic_debug_tstamp_log = []
        self.dynamic_debug_translation_log = []
        self.dynamic_debug_motionmag_log = []
        self.dynamic_debug_out_bound_log = []

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]

        #switching stereo: shift right poses back to left
        right_to_left_bias = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float, device="cuda")
        for t in range(self.counter):
            if t % 2 == 1:
                #right frames
                P_rotation = SO3(torch.tensor([poses[t].data[3].item(), poses[t].data[4].item(), poses[t].data[5].item(), poses[t].data[6].item()], dtype=torch.float, device="cuda"))
                poses[t].data[:3] = poses[t].data[:3] + (P_rotation * right_to_left_bias)

        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        dynamic_slam_logger = {
        'confidence': self.dynamic_debug_confidence_average_log,
        'depth':      self.dynamic_debug_depth_average_log,
        'delta':      self.dynamic_debug_delta_average_log,
        'tstamp':     self.dynamic_debug_tstamp_log,
        'translation':self.dynamic_debug_translation_log,
        'motionmag':  self.dynamic_debug_motionmag_log,
        'outlog':     self.dynamic_debug_out_bound_log
    }


        return poses, tstamps, dynamic_slam_logger

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        #if there was frame deleted, KEYFRAME_INDEX - 1 and KEYFRAME_INDEX + 1 may from different camera
        #if KEYFRAME_INDEX - 1 and KEYFRAME_INDEX + 1 from same camera, use KEYFRAME_INDEX + 1 directly
        #if KEYFRAME_INDEX - 1 and KEYFRAME_INDEX + 1 from different camera, KEYFRAME_INDEX and KEYFRAME_INDEX + 1 must be different, use KEYFRAME_INDEX.
        if self.left_right[self.n - self.cfg.KEYFRAME_INDEX - 1] == self.left_right[self.n - self.cfg.KEYFRAME_INDEX + 1]:
            j = self.n - self.cfg.KEYFRAME_INDEX + 1
        else:
            j = self.n - self.cfg.KEYFRAME_INDEX
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.left_right[i] = self.left_right[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self, depth_only = False):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()
            self.dynamic_debug_current_average_delta = torch.mean(torch.abs(delta[0][self.jj == (self.n-1)])).tolist()
            self.dynamic_debug_current_average_confidence = torch.sum(weight[0][self.jj == (self.n-1)]).tolist()

            #out bound cnt
            total_edges = (corr[0, self.ii == (self.n-1)] == 0).all(dim=1).shape[0] + (corr[0, self.jj == (self.n-1)] == 0).all(dim=1).shape[0]
            total_out = (corr[0, self.ii == (self.n-1)] == 0).all(dim=1).sum().tolist() + (corr[0, self.jj == (self.n-1)] == 0).all(dim=1).sum().tolist()
            self.dynamic_debug_current_out_rate = total_out/total_edges
            

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            if depth_only == True:
                t0 = self.n

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
            self.dynamic_debug_current_average_depth = torch.mean(self.patches_[self.n-1, :, 2, :, :]).tolist()
            dx = (self.poses[0][self.n-1][0] - self.poses[0][self.n-2][0]).tolist()
            dy = (self.poses[0][self.n-1][1] - self.poses[0][self.n-2][1]).tolist()
            dz = (self.poses[0][self.n-1][2] - self.poses[0][self.n-2][2]).tolist()
            dx = dx ** 2
            dy = dy ** 2
            dz = dz ** 2
            self.dynamic_debug_current_translation = math.sqrt(dx + dy + dz)
            self.dynamic_debug_current_motion_mag = self.motionmag(self.n-1, self.n-self.cfg.KEYFRAME_INDEX+1) + self.motionmag(self.n-self.cfg.KEYFRAME_INDEX+1, self.n-1)
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image_left, image_rigt, intrinsics, tstamp_log):
        """ track new frame """
        print("frame :" + str(self.n))
        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')
        
        #switching stereo: self.n % 2 == 0 -> left else -> right
        if self.counter % 2 == 0:
            image = image_left
            self.left_right[self.n] = 0
        else:
            image = image_rigt
            self.left_right[self.n] = 1

        if self.viewer is not None:
            self.viewer.update_image(image)

        image = 2 * (image[None,None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    gradient_bias=self.cfg.GRADIENT_BIAS, 
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        #switching stereo: self.counter % 2 == 0 -> left else -> right
        if self.counter == 1:
            self.poses_[self.n][0] = -0.1

        if self.counter > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])

                xi = (P1 * P2.inv())
                right_to_left_bias = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float, device="cuda") 
                left_to_right_bias = torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float, device="cuda")
                P1_rotation = SO3(torch.tensor([P1.data[3].item(), P1.data[4].item(), P1.data[5].item(), P1.data[6].item()], dtype=torch.float, device="cuda"))
                P2_rotation = SO3(torch.tensor([P2.data[3].item(), P2.data[4].item(), P2.data[5].item(), P2.data[6].item()], dtype=torch.float, device="cuda"))

                if self.left_right[self.n-1] == 1 and self.left_right[self.n-2] == 0:
                    #P1 is right, P2 is left, P1 - P2 has a natual -0.1 bias
                    xi.data[:3] = xi.data[:3] + (P1_rotation * right_to_left_bias) #act on tensor
                elif self.left_right[self.n-1] == 0 and self.left_right[self.n-2] == 1:
                    #P1 is left, P2 is right, P1 - P2 has a natual 0.1 bias
                    xi.data[:3] = xi.data[:3] + (P2_rotation * left_to_right_bias)

                xi = xi.log()
                xi = self.cfg.MOTION_DAMPING * xi
                xi = SE3.exp(xi)
                if self.counter % 2 == 0 and self.left_right[self.n-1] == 1:
                    #current frame is left frame, P1 is a right frame, need a 0.1 bias
                    xi.data[:3] = xi.data[:3] + (P1_rotation * right_to_left_bias)
                elif self.counter % 2 == 1 and self.left_right[self.n-1] == 0:
                    #current frame is right frame, P1 is a left frame, need a -0.1 bias
                    xi.data[:3] = xi.data[:3] + (P1_rotation * left_to_right_bias)

                tvec_qvec = (xi * P1).data

                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        if self.is_initialized == False:
            #let each frame has different depth during init
            patches[:,:,2] = torch.ones_like(patches[:,:,2,0,0,None,None])
        else:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        
        self.n += 1
        self.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True      

            bk_poses = self.poses_[0:10].clone()
            self.update(depth_only = True)

            init_depth = 1
            while self.patches_[0:self.n, :, 2].mean() < init_depth*0.8:
                
                init_depth = init_depth +  1
                self.patches_[0:self.n, :, 2] = torch.ones_like(self.patches_[0:self.n, :, 2])*init_depth
                
                self.poses[0, 0:10] = bk_poses.clone()
                self.update(depth_only = True)

            for itr in range(12):
                self.update()
        
        elif self.is_initialized:
            self.update()
            self.keyframe()

        current_pose = self.poses[0, self.n-1]
        distence = current_pose[0]**2 + current_pose[1]**2 + current_pose[2]**2
        distence = math.sqrt(distence)


        #dynamic slam logger
        if self.is_initialized:
            self.dynamic_debug_delta_average_log.append(self.dynamic_debug_current_average_delta)
            self.dynamic_debug_confidence_average_log.append(self.dynamic_debug_current_average_confidence)
            self.dynamic_debug_depth_average_log.append(self.dynamic_debug_current_average_depth)
            self.dynamic_debug_tstamp_log.append(tstamp_log)
            self.dynamic_debug_translation_log.append(self.dynamic_debug_current_translation)
            self.dynamic_debug_motionmag_log.append(self.dynamic_debug_current_motion_mag)
            self.dynamic_debug_out_bound_log.append(self.dynamic_debug_current_out_rate)
        else:
            self.dynamic_debug_delta_average_log.append(None)
            self.dynamic_debug_confidence_average_log.append(None)
            self.dynamic_debug_depth_average_log.append(None)
            self.dynamic_debug_tstamp_log.append(None)
            self.dynamic_debug_translation_log.append(None)
            self.dynamic_debug_motionmag_log.append(None)
            self.dynamic_debug_out_bound_log.append(None)


            





