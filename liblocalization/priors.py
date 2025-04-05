from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float

from libracecar.batched import batched, batched_zip
from libracecar.numpyro_utils import (
    batched_dist,
    batched_vmap_with_rng,
    prng_key_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast_unchecked_,
    debug_print,
    fval,
    pformat_repr,
)
from libracecar.vector import vec

from .motion import deterministic_position, twist_t


class gaussian(eqx.Module):
    mean: Float[Array, "3"]
    scale_tril: Float[Array, "3 3"]

    __repr__ = pformat_repr

    def covariance(self):
        return self.scale_tril @ self.scale_tril.T

    @staticmethod
    def from_mean_cov(mean: Float[Array, "3"], cov: Float[Array, "3 3"]) -> "gaussian":
        cov_ = jnp.linalg.cholesky(cov)
        return gaussian(mean, cov_)

    def log_prob(self, p: position) -> fval:
        return jnp.array(self.as_dist().log_prob(p.as_arr()))

    def apply_twist(self, twist_p: position):

        def inner(pos_arr: Float[Array, "3"]):
            ans = position.from_arr(pos_arr) + twist_p
            ans_arr = ans.as_arr()
            return ans_arr, ans_arr

        jac, ans_arr = jax.jacfwd(inner, has_aux=True)(self.mean)
        return gaussian.from_mean_cov(ans_arr, jac @ self.covariance() @ jac.T)

    def apply_random_twist(
        self, motion_model: Callable[[], position], n: int
    ) -> "gaussian":
        new_prior = gaussian_mixture.from_gaussian_gen(
            lambda: self.apply_twist(motion_model()), n
        )
        return new_prior.fit_to_gaussian()

    def as_dist(self):
        return dist.MultivariateNormal(
            cast_unchecked_(self.mean), scale_tril=self.scale_tril
        )

    def sample(self) -> position:
        ans = numpyro.sample("draw_particle", self.as_dist())
        return position.from_arr(ans)

    def sample_batch(self, n_points: int) -> batched[position]:
        return vmap_seperate_seed(
            lambda: batched.create(self.sample()),
            axis_size=n_points,
        )()

    def plot(self, n_points: int, style: plot_style = plot_style()) -> plotable:
        return self.sample_batch(n_points).map(lambda x: x.plot_as_seg(style))


class gaussian_mixture(eqx.Module):
    parts: batched[tuple[gaussian, fval]]

    __repr__ = pformat_repr

    @property
    def mean_buf(self):
        return self.parts.unflatten()[0].mean

    @property
    def scale_tril_buf(self):
        return self.parts.unflatten()[0].scale_tril

    @property
    def weights_buf(self):
        return self.parts.unflatten()[1]

    @staticmethod
    def one(g: gaussian):
        return gaussian_mixture(batched.create((g, jnp.array(1.0))).reshape(1))

    def as_dist(self) -> dist.Distribution:
        _, ws = self.parts.split_tuple()
        parts = self.parts.tuple_map(lambda x, w: x.as_dist())

        mixing_distribution = dist.CategoricalProbs(probs=ws.unflatten())

        # parts.map(lambda g: (g.mean, g.scale_tril))

        # return dist.MixtureSameFamily(mixing_distribution, parts.unflatten())
        return dist.MixtureSameFamily(mixing_distribution, batched_dist(parts))

    def sample(self) -> position:
        ans = numpyro.sample("draw_particle", self.as_dist())
        return position.from_arr(ans)

    def sample_batch(self, n_points: int) -> batched[position]:
        return vmap_seperate_seed(
            lambda: batched.create(self.sample()),
            axis_size=n_points,
        )()

    def mean(self) -> position:
        return position.from_arr(self.as_dist().mean)

    def fit_to_gaussian(self):
        ans_mean = self.parts.tuple_map(lambda g, w: g.mean * w).sum().unwrap()
        ans_cov = (
            self.parts.tuple_map(
                lambda g, w: (
                    w
                    * (g.covariance() + jnp.outer(g.mean - ans_mean, g.mean - ans_mean))
                )
            )
            .sum()
            .unwrap()
        )
        return gaussian.from_mean_cov(ans_mean, ans_cov)

    @staticmethod
    def from_gaussian_gen(fn: Callable[[], gaussian], n: int):
        ans = vmap_seperate_seed(
            lambda: batched.create((fn(), jnp.array(1 / n))), axis_size=n
        )()
        return gaussian_mixture(ans)

    def apply_random_twist(self, motion_model: Callable[[], position], n: int):
        def inner(v: tuple[gaussian, fval]):
            g, w = v
            return g.apply_random_twist(motion_model, n), w

        ans = batched_vmap_with_rng(inner, self.parts)
        return gaussian_mixture(ans)

    def plot(self, n_points: int, style: plot_style = plot_style()) -> plotable:
        return self.sample_batch(n_points).map(lambda x: x.plot_as_seg(style))

    def resample(self, count: int, keep: int) -> "gaussian_mixture":
        assert keep < count
        redraw = count - keep

        parts = self.parts.sort(lambda x: x[1])

        stay_parts = parts[redraw:]

        redraw_parts, ws = parts[:redraw].split_tuple()
        redraw_weight = ws.sum().unwrap()
        debug_print("redraw_weight", redraw_weight)

        # ws = ws.map(lambda x: x / redraw_weight)

        each_cov = self.fit_to_gaussian().covariance() / redraw / 2.0

        means = self.sample_batch(redraw)
        ans = means.map(
            lambda x: (
                gaussian.from_mean_cov(x.as_arr(), each_cov),
                jnp.array(redraw_weight / redraw),
            ),
        )
        ret = gaussian_mixture(batched.concat([ans, stay_parts]))
        # debug_print("ret", ret)
        return ret


class particles(eqx.Module):
    # (pose, weight)
    points: batched[tuple[position, fval]]

    __repr__ = pformat_repr

    def cat_dist(self) -> dist.Distribution:
        _, probs = self.points.split_tuple()
        return dist.CategoricalProbs(probs=probs.unflatten())

    def sample(self) -> position:
        ans = numpyro.sample("draw_particle", self.cat_dist())
        pos, _ = self.points.split_tuple()
        return pos[ans].unwrap()

    @staticmethod
    def from_samples(s: batched[position]):
        (n,) = s.batch_dims()
        return particles(s.map(lambda x: (x, jnp.array(1 / n))))

    @staticmethod
    def from_logits(s: batched[tuple[position, fval]]):
        (n,) = s.batch_dims()
        pos, ws = s.split_tuple()
        ws = batched.create(jax.nn.softmax(ws.unflatten()), batch_dims=(n,))
        return particles(batched_zip(pos, ws))

    def resample(self, count: int) -> "particles":
        ans = vmap_seperate_seed(
            lambda: batched.create(self.sample()),
            axis_size=count,
        )()
        return particles.from_samples(ans)

    def map(self, fn: Callable[[position], position]) -> "particles":
        ans = batched_vmap_with_rng(lambda x: (fn(x[0]), x[1]), self.points)
        return particles(ans)

    # def max_likelyhood(self) -> position:
    #     pos, probs = self.points.split_tuple()
    #     best = jnp.argmax(probs.unflatten())
    #     return pos[best].unwrap()

    def mean(self) -> position:
        tran, rot = (
            self.points.tuple_map(lambda p, w: (p.tran * w, vec(p.rot._v) * w))
            .sum()
            .unwrap()
        )
        return position(tran, rot.normalize())

    def fit_to_gaussian(self):
        arrs, ws = self.points.tuple_map(lambda p, w: (p.as_arr(), w)).split_tuple()
        cov = jnp.cov(arrs.unflatten(), aweights=ws.unflatten(), rowvar=False)
        assert cov.shape == (3, 3)
        return gaussian.from_mean_cov(
            batched_zip(arrs, ws).tuple_map(lambda x, w: x * w).sum().unwrap(), cov
        )

    def plot(self, n_points: int, style: plot_style = plot_style()) -> plotable:
        return self.resample(n_points).points.tuple_map(
            lambda x, _: x.plot_as_seg(style)
        )
