# coding: utf-8

"""
Motion extractor(M), which directly predicts the canonical keypoints, head pose and expression deformation of the input image
"""

from torch import nn
import torch
from diffusers.models.modeling_utils import ModelMixin
from .convnextv2 import convnextv2_tiny
from .liveportrait_util import filter_state_dict
from .camera import headpose_pred_to_degree, get_rotation_matrix

model_dict = {
    'convnextv2_tiny': convnextv2_tiny,
}


class MotionExtractor(ModelMixin):
    def __init__(self, **kwargs):
        super(MotionExtractor, self).__init__()

        # default is convnextv2_base
        backbone = kwargs.get('backbone', 'convnextv2_tiny')
        self.detector = model_dict.get(backbone)(**kwargs)
        self.register_buffer('idx_tensor', torch.arange(66, dtype=torch.float32))

    def headpose_pred_to_degree(self, pred):
        """
        pred: (bs, 66) or (bs, 1) or others
        """
        if pred.ndim > 1 and pred.shape[1] == 66:
            # NOTE: note that the average is modified to 97.5
            prob = torch.nn.functional.softmax(pred, dim=1)
            degree = torch.matmul(prob, self.idx_tensor)
            degree = degree * 3 - 97.5

            return degree

        return pred

    def load_pretrained(self, init_path: str):
        if init_path not in (None, ''):
            state_dict = torch.load(init_path, map_location=lambda storage, loc: storage)['model']
            state_dict = filter_state_dict(state_dict, remove_name='head')
            ret = self.detector.load_state_dict(state_dict, strict=False)
            print(f'Load pretrained model from {init_path}, ret: {ret}')

    def forward(self, x):
        kp_info = self.detector(x)
        return self.get_kp(kp_info)

    def get_kp(self, kp_info):
        bs = kp_info['kp'].shape[0]

        angles_raw = torch.cat([kp_info['pitch'], kp_info['yaw'], kp_info['roll']], dim=0) # (3, 66)
        angles_deg = self.headpose_pred_to_degree(angles_raw)[:, None] # (B, 3)
        pitch, yaw, roll = torch.chunk(angles_deg, chunks=3, dim=0)


        kp = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        t, scale = kp_info['t'], kp_info['scale']

        rot_mat = get_rotation_matrix(pitch, yaw, roll).to(self.dtype)    # (bs, 3, 3)

        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        # Eqn.2: s * (R * x_c,s) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat# + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    def interpolate_tensors(self, a: torch.Tensor, b: torch.Tensor, num: int = 10) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

        B, *rest = a.shape
        alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
        view_shape = (num,) + (1,) * len(rest)
        alphas = alphas.view(view_shape)  # (1, num, 1, 1, ...)

        result = (1 - alphas) * a + alphas * b
        return result[:-1]

    def interpolate_kps(self, ref, motion, num_interp, t_scale=0.5, s_scale=0):
        kp1 = self.detector(ref.to(self.dtype))
        kp2_list = []
        for i in range(0, motion.shape[0], 256):
            motion_chunk = motion[i:i+256]
            kp2_chunk = self.detector(motion_chunk.to(self.dtype))
            kp2_list.append(kp2_chunk)
        kp2 = {}
        for key in kp2_list[0].keys():
            kp2[key] = torch.cat([kp2_chunk[key] for kp2_chunk in kp2_list], dim=0)

        angles_raw = torch.cat([kp1['pitch'], kp1['yaw'], kp1['roll']], dim=0) # (3, 66)
        angles_deg = self.headpose_pred_to_degree(angles_raw) # (B, 3)
        pitch_1, yaw_1, roll_1 = torch.chunk(angles_deg, chunks=3, dim=0)

        angles_raw = torch.cat([kp2['pitch'], kp2['yaw'], kp2['roll']], dim=0) # (3, 66)
        angles_deg = self.headpose_pred_to_degree(angles_raw) # (B, 3)
        pitch_2, yaw_2, roll_2 = torch.chunk(angles_deg, chunks=3, dim=0)

        pitch_interp = self.interpolate_tensors(pitch_1, pitch_2[:1], num_interp)  # Bx(num_interp)x1
        yaw_interp = self.interpolate_tensors(yaw_1, yaw_2[:1], num_interp) # Bx(num_interp)x1
        roll_interp = self.interpolate_tensors(roll_1, roll_2[:1], num_interp)  # Bx(num_interp)x1

        t_1 = kp1['t']
        t_2 = kp2['t']
        t_2 = (t_2 - t_2[0]) * t_scale + t_1
        t_interp = self.interpolate_tensors(t_1, t_2[:1], num_interp)

        s_1 = kp1['scale']
        s_2 = kp2['scale']
        s_2 = s_2 * s_scale + s_1
        s_interp = self.interpolate_tensors(s_1, s_2[:1], num_interp)

        kp = kp1['kp'].repeat(num_interp+motion.shape[0]-1, 1)

        kps_interp = {
            'pitch': torch.cat([pitch_interp, pitch_2], dim=0),
            'yaw': torch.cat([yaw_interp, yaw_2], dim=0),
            'roll': torch.cat([roll_interp, roll_2], dim=0),
            't': torch.cat([t_interp, t_2], dim=0),
            'scale': torch.cat([s_interp, s_2], dim=0),
            'kp': kp
        }

        kp_intrep = self.get_kp(kps_interp)

        return kp_intrep


    def interpolate_kps_online(self, ref, motion, num_interp, t_scale=0.5, s_scale=0):
        kp1 = self.detector(ref.to(self.dtype))
        kp_frame1 = self.detector(motion[:1].to(self.dtype))
        kp2 = self.detector(motion.to(self.dtype))

        angles_raw = torch.cat([kp1['pitch'], kp1['yaw'], kp1['roll']], dim=0) # (3, 66)
        angles_deg = self.headpose_pred_to_degree(angles_raw) # (B, 3)
        pitch_1, yaw_1, roll_1 = torch.chunk(angles_deg, chunks=3, dim=0)

        angles_raw = torch.cat([kp2['pitch'], kp2['yaw'], kp2['roll']], dim=0) # (3, 66)
        angles_deg = self.headpose_pred_to_degree(angles_raw) # (B, 3)
        pitch_2, yaw_2, roll_2 = torch.chunk(angles_deg, chunks=3, dim=0)

        pitch_interp = self.interpolate_tensors(pitch_1, pitch_2[:1], num_interp)  # Bx(num_interp)x1
        yaw_interp = self.interpolate_tensors(yaw_1, yaw_2[:1], num_interp) # Bx(num_interp)x1
        roll_interp = self.interpolate_tensors(roll_1, roll_2[:1], num_interp)  # Bx(num_interp)x1

        t_1 = kp1['t']
        t_2 = kp2['t']
        t_2 = (t_2 - t_2[0]) * t_scale + t_1
        t_interp = self.interpolate_tensors(t_1, t_2[:1], num_interp)

        s_1 = kp1['scale']
        s_2 = kp2['scale']
        s_2 = s_2 * s_scale + s_1
        s_interp = self.interpolate_tensors(s_1, s_2[:1], num_interp)

        kp = kp1['kp'].repeat(num_interp+motion.shape[0]-1, 1)

        kps_interp = {
            'pitch': torch.cat([pitch_interp, pitch_2], dim=0),
            'yaw': torch.cat([yaw_interp, yaw_2], dim=0),
            'roll': torch.cat([roll_interp, roll_2], dim=0),
            't': torch.cat([t_interp, t_2], dim=0),
            'scale': torch.cat([s_interp, s_2], dim=0),
            'kp': kp
        }

        kp_intrep = self.get_kp(kps_interp)

        kp_dri = self.get_kp(kp2)

        return kp_intrep, kp1, kp_frame1, kp_dri

    def get_kps(self, kp_ref, kp_frame1, motion, t_scale=0.5, s_scale=0):
        kps_motion = self.detector(motion.to(self.dtype))

        kps_dri = self.get_kp(kps_motion)

        t_ref = kp_ref['t']
        t_frame1 = kp_frame1['t']
        t_motion = kps_motion['t']
        kps_motion['t'] = (t_motion - t_frame1) * t_scale + t_ref

        s_ref = kp_ref['scale']
        s_motion = kps_motion['scale']
        kps_motion['scale'] = s_motion * s_scale + s_ref


        kps_motion['kp'] = kp_ref['kp'].repeat(motion.shape[0], 1)

        kps_motion = self.get_kp(kps_motion)

        return kps_motion, kps_dri

    def inference(self, ref, motion):
        kps_ref = self.detector(ref.to(self.dtype))
        kps_motion = self.detector(motion.to(self.dtype))
        kps_motion['kp'] = kps_ref['kp']

        kp_s = self.get_kp(kps_ref)
        kp_d = self.get_kp(kps_motion)

        return kp_s, kp_d
