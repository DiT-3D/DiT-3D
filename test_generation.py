import torch
from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *

from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds

from models.dit3d import DiT3D_models

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, y, clip_denoised: bool, return_pred_xstart: bool):
        
        # print('data:', data.shape)
        model_output = denoise_fn(data, t, y)
        # print('model_output:', model_output.shape)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, y, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, y=y, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, y,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def reconstruct(self, x0, t, y, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        # DiT-3d
        self.model = DiT3D_models[args.model_type](input_size=args.voxel_size, num_classes=args.num_classes)


    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, y, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, y)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None, y=None):
        B, D, N = data.shape                           # [16, 3, 2048]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset



def evaluate_gen(opt, ref_pcs, logger):

    if ref_pcs is None:
        _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category, use_mask=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False)
        ref = []
        for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
            x = data['test_points']
            m, s = data['mean'].float(), data['std'].float()

            ref.append(x*s + m)

        ref_pcs = torch.cat(ref, dim=0).contiguous()

    logger.info("Loading sample path: %s"
      % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    logger.info("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))


    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, opt.batch_size)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}

    pprint(results)
    logger.info(results)

    jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
    pprint('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))



def generate(model, opt):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)


    def new_y_chain(device, num_chain, num_classes):
        return torch.randint(low=0,high=num_classes,size=(num_chain,),device=device)
    
    with torch.no_grad():

        samples = []
        ref = []

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            x = data['test_points'].transpose(1,2)
            m, s = data['mean'].float(), data['std'].float()

            y = data['cate_idx']

            gen = model.gen_samples(x.shape, 'cuda', new_y_chain('cuda',y.shape[0],opt.num_classes), clip_denoised=False).detach().cpu()

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()



            gen = gen * s + m
            x = x * s + m
            samples.append(gen)
            ref.append(x)

            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), f'{i}.png'), gen[:64], None,
                                       None, None)

        samples = torch.cat(samples, dim=0)
        ref = torch.cat(ref, dim=0)

        torch.save(samples, opt.eval_path)



    return ref


def main(opt):

    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.cuda:
        model.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param['model_state'])


        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref=generate(model, opt)
            
        if opt.eval_gen:
            # Evaluate generation
            evaluate_gen(opt, ref, logger)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=55)

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--eval_gen', action='store_true', default=False)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument("--voxel_size", type=int, choices=[16, 32, 64], default=32)
    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-XL/2")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')


    parser.add_argument('--model', default='',required=True, help="path to model (to continue training)")

    '''eval'''

    parser.add_argument('--eval_path',
                        default='')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
