from torch.optim.optimizer import Optimizer
import torch
import math
import numpy as np

class DemonRanger(Optimizer):

    # Rectified-AMSGrad/RAdam + AdaMod + QH Momentum + Iterat Averaging + Lookahead + DEMON (decaying Momentum) + gradient centralization + grad noise

    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999, 0.999),
                 nus=(0.7, 1.0),
                 eps=1e-8,
                 k=5,
                 alpha=0.8,
                 gamma=0.55,
                 use_demon=True,
                 rectify=True,
                 amsgrad=True,
                 AdaMod=True,
                 AdaMod_bias_correct=True,
                 IA=True,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0,
                 use_gc=True,
                 use_grad_noise=False):

        # betas = (beta1 for first order moments, beta2 for second order moments, beta3 for ema over adaptive learning rates (AdaMod))
        # nus = (nu1,nu2) (for quasi hyperbolic momentum)
        # eps = small value for numerical stability (avoid divide by zero)
        # k = lookahead cycle
        # alpha = outer learning rate (lookahead)
        # gamma = gradient noise control parameter (for regularization)
        # use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        # rectify = bool to decide whether to apply the recitification term (from RAdam) or not
        # amsgrad = bool to decide whether to use amsgrad instead of adam as the core optimizer
        # AdaMod_bias_correct = bool to decide whether to add bias correction to AdaMod
        # IA = bool to decide if Iterate Averaging is ever going to be used
        # IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        # epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        # step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        # weight decay = decorrelated weight decay value
        # use_gc = bool to determine whether to use gradient centralization or not.
        # use_grad_noise = bool to determine whether to use gradient noise or not.

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError("Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError("Invalid nu parameter at index 1: {}".format(nus[1]))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        self.use_gc = use_gc
        self.use_grad_noise = use_grad_noise
        self.k = k
        self.epochs = epochs
        self.amsgrad = amsgrad
        self.use_demon = use_demon
        self.IA_cycle = IA_cycle
        self.IA = IA
        self.rectify = rectify
        self.AdaMod = AdaMod
        self.AdaMod_bias_correct = AdaMod_bias_correct
        if step_per_epoch is None:
            self.step_per_epoch = IA_cycle
        else:
            self.step_per_epoch = step_per_epoch

        self.T = self.epochs*self.step_per_epoch

        defaults = dict(lr=lr,
                        betas=betas,
                        nus=nus,
                        eps=eps,
                        alpha=alpha,
                        gamma=gamma,
                        weight_decay=weight_decay)
        super(DemonRanger, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DemonRanger, self).__setstate__(state)

    def apply_AdaMod(self, beta3, n_avg, n, step):
        n_avg.mul_(beta3).add_(n, alpha = 1 - beta3)
        if self.AdaMod_bias_correct:
            n_avg_ = n_avg.clone()
            n_avg_.div_(1 - (beta3 ** step))
            torch.min(n, n_avg_, out=n)
        else:
            torch.min(n, n_avg, out=n)
        return n

    def step(self, activate_IA=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('DemonRanger does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['num_models'] = 0
                    state['cached_params'] = p.data.clone()
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.AdaMod:
                        state['n_avg'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1_init, beta2, beta3 = group['betas']
                rho_inf = (2/(1-beta2)) - 1
                nu1, nu2 = group['nus']
                lr = group['lr']
                wd = group['weight_decay']
                alpha = group['alpha']
                gamma = group['gamma']

                do_IA = False
                lookahead_step = False

                if self.IA and activate_IA:
                    lookahead_step = False
                    if state['step'] % self.IA_cycle == 0:
                        do_IA = True
                elif self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if self.use_demon:
                    temp = 1-(state['step']/self.T)
                    beta1 = beta1_init * temp / ((1-beta1_init)+beta1_init*temp)
                else:
                    beta1 = beta1_init

                if self.use_grad_noise:
                    grad_var = lr/((1+state['step'])**gamma)
                    grad_noise = torch.empty_like(grad).normal_(mean=0.0, std=math.sqrt(grad_var))
                    grad.add_(grad_noise)

                if self.use_gc and grad.view(-1).size(0) > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)

                momentum = exp_avg.clone()
                momentum.div_(1 - (beta1 ** state['step'])).mul_(nu1).add_(grad, alpha = 1-nu1)

                if wd != 0:
                    p.data.add_(p.data, alpha = -wd*lr)

                beta2_t = beta2 ** state['step']

                if self.amsgrad and state['step'] > 1:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    vt = max_exp_avg_sq.clone()
                else:
                    vt = exp_avg_sq.clone()

                if self.rectify:
                    rho_t = rho_inf - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    # more conservative since it's an approximated value
                    if rho_t >= 5:
                        R = math.sqrt(((rho_t-4)*(rho_t-2)*rho_inf) /
                                      ((rho_inf-4)*(rho_inf-2)*rho_t))
                        bias_correction2 = 1 - beta2_t
                        vt.div_(bias_correction2)
                        if nu2 != 1.0:
                            vt.mul_(nu2).addcmul_(grad, grad, value = 1-nu2)
                        denom = vt.sqrt_().add_(group['eps'])

                        n = (lr*R)/denom

                        if self.AdaMod:
                            n_avg = state['n_avg']
                            n = self.apply_AdaMod(beta3, n_avg, n, step=state['step'])

                        p.data.add_(-n*momentum)
                    else:
                        if self.AdaMod:
                            n_avg = state['n_avg']
                            n_avg.mul_(beta3).add_(lr, alpha = 1 - beta3)
                        p.data.add_(momentum, alpha = -lr)
                else:
                    bias_correction2 = 1 - beta2_t
                    vt.div_(bias_correction2)
                    if nu2 != 1.0:
                        vt.mul_(nu2).addcmul_(grad, grad, value = 1-nu2)
                    denom = vt.sqrt_().add_(group['eps'])
                    n = lr/denom
                    if self.AdaMod:
                        n_avg = state['n_avg']
                        n = self.apply_AdaMod(beta3, n_avg, n, step=state['step'])

                    p.data.add_(-n*momentum)

                if lookahead_step:
                    p.data.mul_(alpha).add_(state['cached_params'], alpha = 1.0 - alpha)
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"]+1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss