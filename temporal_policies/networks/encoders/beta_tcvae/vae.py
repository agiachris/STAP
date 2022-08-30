"""Implementation from https://github.com/rtqichen/beta-tcvae."""

import argparse
import math
import os
import time

import torch

from . import datasets, dist, elbo_decomposition, flows, utils


class MLPEncoder(torch.nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = torch.nn.Linear(4096, 1200)
        self.fc2 = torch.nn.Linear(1200, 1200)
        self.fc3 = torch.nn.Linear(1200, output_dim)

        self.conv_z = torch.nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1200),
            torch.nn.Tanh(),
            torch.nn.Linear(1200, 1200),
            torch.nn.Tanh(),
            torch.nn.Linear(1200, 1200),
            torch.nn.Tanh(),
            torch.nn.Linear(1200, 4096),
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(torch.nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = torch.nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 512, 4)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.conv_z = torch.nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = torch.nn.BatchNorm2d(512)
        self.conv2 = torch.nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.conv5 = torch.nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.conv_final = torch.nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        use_cuda=False,
        prior_dist=dist.Normal(),
        q_dist=dist.Normal(),
        include_mutinfo=True,
        tcvae=False,
        conv=False,
        mss=False,
        encoder=None,
        decoder=None,
    ):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer("prior_params", torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if encoder is not None:
            if decoder is None:
                raise ValueError(
                    "Encoder and decoder must both be None or both be not None."
                )
            self.encoder = encoder
            self.decoder = decoder
        elif conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = torch.autograd.Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        # x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(
            x.size(0), self.z_dim, self.q_dist.nparams
        )
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        # x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        x_params = self.decoder.forward(z)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        # x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = (
            self.prior_dist.log_density(zs, params=prior_params)
            .view(batch_size, -1)
            .sum(1)
        )
        logqz_condx = (
            self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        )

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach(), x_recon.detach(), zs.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams),
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (
                utils.logsumexp(_logqz, dim=1, keepdim=False)
                - math.log(batch_size * dataset_size)
            ).sum(1)
            logqz = utils.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(
                batch_size * dataset_size
            )
        else:
            # minibatch stratified sampling
            logiw_matrix = torch.autograd.Variable(
                self._log_importance_weight_matrix(batch_size, dataset_size).type_as(
                    _logqz.data
                )
            )
            logqz = utils.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = utils.logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz,
                dim=1,
                keepdim=False,
            ).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) - self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals)
                    + (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = (
                    logpx
                    - (logqz_condx - logqz)
                    - self.beta * (logqz - logqz_prodmarginals)
                    - (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = (
                    logpx
                    - self.beta * (logqz - logqz_prodmarginals)
                    - (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )

        return modified_elbo, elbo.detach(), x_recon.detach(), zs.detach()


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == "shapes":
        train_set = datasets.Shapes()
    elif args.dataset == "faces":
        train_set = datasets.Faces()
    else:
        raise ValueError("Unknown dataset " + str(args.dataset))

    kwargs = {"num_workers": 4, "pin_memory": use_cuda}
    train_loader = torch.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    return train_loader


def anneal_kl(args, vae, iteration):
    if args.dataset == "shapes":
        warmup_iter = 7000
    elif args.dataset == "faces":
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main(args):

    torch.cuda.set_device(args.gpu)

    # data loader
    train_loader = setup_data_loaders(args, use_cuda=True)

    # setup the VAE
    if args.dist == "normal":
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == "laplace":
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == "flow":
        prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(
        z_dim=args.latent_dim,
        use_cuda=True,
        prior_dist=prior_dist,
        q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo,
        tcvae=args.tcvae,
        conv=args.conv,
        mss=args.mss,
    )

    # setup the optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            # x = x.cuda(async=True)
            # wrap the mini-batch in a PyTorch Variable
            x = torch.autograd.Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError("NaN spotted in objective.")
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().data[0])
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print(
                    "[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)"
                    % (
                        iteration,
                        time.time() - batch_time,
                        vae.beta,
                        vae.lamb,
                        elbo_running_mean.val,
                        elbo_running_mean.avg,
                    )
                )

                vae.eval()

                utils.save_checkpoint(
                    {"state_dict": vae.state_dict(), "args": args}, args.save, 0
                )
                eval("plot_vs_gt_" + args.dataset)(
                    vae,
                    train_loader.dataset,
                    os.path.join(
                        args.save, "gt_vs_latent_{:05d}.png".format(iteration)
                    ),
                )

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({"state_dict": vae.state_dict(), "args": args}, args.save, 0)
    dataset_loader = torch.dataset.DataLoader(
        train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False
    )
    (
        logpx,
        dependence,
        information,
        dimwise_kl,
        analytical_cond_kl,
        marginal_entropies,
        joint_entropy,
    ) = elbo_decomposition.elbo_decomposition(vae, dataset_loader)
    torch.save(
        {
            "logpx": logpx,
            "dependence": dependence,
            "information": information,
            "dimwise_kl": dimwise_kl,
            "analytical_cond_kl": analytical_cond_kl,
            "marginal_entropies": marginal_entropies,
            "joint_entropy": joint_entropy,
        },
        os.path.join(args.save, "elbo_decomposition.pth"),
    )
    eval("plot_vs_gt_" + args.dataset)(
        vae, dataset_loader.dataset, os.path.join(args.save, "gt_vs_latent.png")
    )
    return vae


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-d",
        "--dataset",
        default="shapes",
        type=str,
        help="dataset name",
        choices=["shapes", "faces"],
    )
    parser.add_argument(
        "-dist", default="normal", type=str, choices=["normal", "laplace", "flow"]
    )
    parser.add_argument(
        "-n", "--num-epochs", default=50, type=int, help="number of training epochs"
    )
    parser.add_argument("-b", "--batch-size", default=2048, type=int, help="batch size")
    parser.add_argument(
        "-l", "--learning-rate", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "-z", "--latent-dim", default=10, type=int, help="size of latent dimension"
    )
    parser.add_argument("--beta", default=1, type=float, help="ELBO penalty term")
    parser.add_argument("--tcvae", action="store_true")
    parser.add_argument("--exclude-mutinfo", action="store_true")
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--lambda-anneal", action="store_true")
    parser.add_argument(
        "--mss", action="store_true", help="use the improved minibatch estimator"
    )
    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--visdom", action="store_true", help="whether plotting in visdom is desired"
    )
    parser.add_argument("--save", default="test1")
    parser.add_argument(
        "--log_freq", default=200, type=int, help="num iterations per log"
    )
    args = parser.parse_args()
    main(args)
