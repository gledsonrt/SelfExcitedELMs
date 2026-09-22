"""Analytical flat-plate forcing and data utilities."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import hankel2
from scipy.stats import qmc


def _theodorsen(k):
    k = np.asarray(k, float)
    # For real arguments, the second-kind Hankel form gives Theodorsen's
    # circulation function directly.
    return hankel2(1, k) / (hankel2(1, k) + 1j * hankel2(0, k))


def flat_plate_derivatives(reduced_velocity, include_apparent_mass: bool = True) -> np.ndarray:
    """Return `[H1, H2, H3, H4, A1, A2, A3, A4]` from Theodorsen theory."""
    vr = np.atleast_1d(np.asarray(reduced_velocity, float))
    k, K = np.pi / vr, 2 * np.pi / vr
    c = _theodorsen(k); f, g = c.real, c.imag
    if include_apparent_mass:
        values = (-2*np.pi*f/K, -np.pi*(1 + 4*g/K + f)/(2*K),
                  -np.pi*(2*f - .5*g*K)/K**2, .5*np.pi*(1 + 4*g/K),
                  .5*np.pi*f/K, -np.pi*(K/4 - g - K*f/4)/(2*K**2),
                  .5*np.pi*(K**2/32 + f - K*g/4)/K**2, -.5*np.pi*g/K)
    else:
        values = (-2*np.pi*f/K, -2*np.pi*g/K**2, -2*np.pi*f/K**2,
                  2*np.pi*g/K, .5*np.pi*f/K, .5*np.pi*g/K**2,
                  .5*np.pi*f/K**2, -.5*np.pi*g/K)
    result = np.column_stack(values)
    return result[0] if np.ndim(reduced_velocity) == 0 else result


def _harmonic_motion(rho, u, b, vr, amplitude, dt, n_steps, pitch=False):
    omega = 2*np.pi*u/(b*vr); k = omega*b/(2*u); c = _theodorsen(k)
    time = np.arange(n_steps + 1) * dt
    phase = amplitude * np.exp(1j * omega * time - 1j*np.pi/2)
    velocity, acceleration = 1j*omega*phase, (1j*omega)**2*phase
    if not pitch:
        lift = -np.pi*rho*b**2*acceleration/4 - .5*rho*u**2*b*2*np.pi*c*velocity/u
        moment = .5*rho*u**2*b**2*np.pi*c*velocity/(2*u)
    else:
        lift = -np.pi*rho*b**2*u*velocity/4 - .5*rho*u**2*b*2*np.pi*c*(phase + .25*b*velocity/u)
        moment = (-np.pi*rho*b**3*(u*velocity + b*acceleration/8)/16
                  + .5*rho*u**2*b**2*np.pi*c*(phase + .25*b*velocity/u)/2)
    return tuple(np.real(v) for v in (lift, moment, time, phase, velocity, acceleration))


def harmonic_heave(rho, u, b, vr, amplitude, dt, n_steps):
    return _harmonic_motion(rho, u, b, vr, amplitude, dt, n_steps, False)


def harmonic_pitch(rho, u, b, vr, amplitude, dt, n_steps):
    return _harmonic_motion(rho, u, b, vr, amplitude, dt, n_steps, True)


@dataclass
class FlatPlateDataset:
    h: np.ndarray; hdot: np.ndarray; hddot: np.ndarray; a: np.ndarray; adot: np.ndarray; addot: np.ndarray
    h_vr: np.ndarray; a_vr: np.ndarray
    flh: np.ndarray; fmh: np.ndarray; fla: np.ndarray; fma: np.ndarray; amp_h: float; amp_a: float


def generate_flat_plate_dataset(rho=1.2, u=20., b=31., dt=.00775, vrs=None,
                                amplitudes=None, steps_per_cycle=25, n_cycles=1,
                                snr=np.inf, random_state=0) -> FlatPlateDataset:
    """Return Latin-hypercube-sampled, normalized harmonic training data."""
    vrs = np.asarray(vrs if vrs is not None else np.r_[np.linspace(.5, 20, 20), np.geomspace(.25, 20, 20)])
    amps = np.asarray(amplitudes if amplitudes is not None else np.c_[np.linspace(np.finfo(float).eps, .85, 50), np.linspace(np.finfo(float).eps, .12, 50)])
    if amps.ndim == 1: amps = np.c_[amps, amps]
    if vrs.ndim == 1: vrs = np.c_[vrs, vrs]
    count = len(vrs) * len(amps) * steps_per_cycle * n_cycles
    data = [np.empty(count) for _ in range(12)]; cursor = 0
    rng = np.random.default_rng(random_state)
    for amp_h, amp_a in amps:
        for vr_h, vr_a in vrs:
            size = steps_per_cycle*n_cycles
            for motion, vr, amp, input_offset, velocity_offset, force_offset in ((harmonic_heave, vr_h, amp_h, 0, 6, 8), (harmonic_pitch, vr_a, amp_a, 3, 7, 10)):
                n = n_cycles * int(np.ceil((b*vr/u) / dt))
                l, m, _, q, qdot, qddot = motion(rho, u, b, vr, amp, dt, n)
                sample = np.minimum((qmc.LatinHypercube(1, seed=rng.integers(2**32)).random(size)[:, 0] * n).astype(int), n)
                # Retain space-filling samples, but explicitly include the
                # displacement zero crossings and extrema.  These states are
                # important for force prediction and can be missed by a small
                # Latin-hypercube draw.
                anchors = np.rint(np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * n).astype(int)
                sample[:min(len(anchors), size)] = anchors[:min(len(anchors), size)]
                sample = np.sort(sample)
                data[input_offset][cursor:cursor+size], data[input_offset+1][cursor:cursor+size], data[input_offset+2][cursor:cursor+size] = q[sample], qdot[sample], qddot[sample]
                data[velocity_offset][cursor:cursor+size] = vr
                data[force_offset][cursor:cursor+size], data[force_offset+1][cursor:cursor+size] = l[sample]/(.5*rho*b*u**2), m[sample]/(.5*rho*b**2*u**2)
            cursor += size
    for force in data[8:12]: force += force.std() * rng.standard_normal(count) / snr
    return FlatPlateDataset(*data, amp_h=float(amps[-1, 0]), amp_a=float(amps[-1, 1]))


def get_derivatives(u, b, rho, vr, lh, mh, lp, mp, dh, vh, dp, vp):
    """Identify the eight flutter derivatives with least squares."""
    k = 2*np.pi/vr
    def fit(y, columns): return np.linalg.lstsq(np.column_stack(columns), np.asarray(y), rcond=None)[0]
    h1, h4 = fit(lh, [-.5*rho*u**2*b*(-np.asarray(vh))*k/u, -.5*rho*u**2*b*(-np.asarray(dh))*k**2/b])
    a1, a4 = fit(mh, [.5*rho*u**2*b**2*(-np.asarray(vh))*k/u, .5*rho*u**2*b**2*(-np.asarray(dh))*k**2/b])
    h2, h3 = fit(lp, [-.5*rho*u**2*b**2*np.asarray(vp)*k/u, -.5*rho*u**2*b*np.asarray(dp)*k**2])
    a2, a3 = fit(mp, [.5*rho*u**2*b**3*np.asarray(vp)*k/u, .5*rho*u**2*b**2*np.asarray(dp)*k**2])
    return np.array([h1, h2, h3, h4, a1, a2, a3, a4])


def _convolve(a, b, dt):
    n = len(a); return np.fft.ifft(np.fft.fft(np.r_[a, np.zeros(n)]) * np.fft.fft(np.r_[b, np.zeros(n)])).real[:n] * dt


def unsteady_features(u, b, dt, h, a):
    """Return causal Wagner-history features for heave and pitch force models.

    These are the physical state features required to model non-harmonic
    motion.  They are calculated from the motion alone, with no force input.
    """
    h, a = np.asarray(h, float) - np.mean(h), np.asarray(a, float) - np.mean(a)
    time = np.arange(1, len(h)+1) * dt
    tau, ds = time*u/b, dt*u/b
    hdot, adot = np.gradient(h, dt), np.gradient(a, dt)
    hp = np.gradient(hdot, dt)*b**2/u**2
    ap, app = adot*b/u, np.gradient(adot, dt)*b**2/u**2
    phi_se = 1-.165*np.exp(-.089*tau)-.335*np.exp(-.6*tau)
    se_h = _convolve(phi_se, hp/b, ds)
    se_a = _convolve(phi_se, ap, ds) + _convolve(phi_se, app, ds/4)
    features_h, features_a = np.c_[hp, se_h], np.c_[ap, app, se_a]
    # `unsteady_time_domain` removes the mean from each force component.
    # Centering each causal feature sequence preserves that convention when
    # fitting force coefficients across multiple independently generated runs.
    return features_h - features_h.mean(axis=0), features_a - features_a.mean(axis=0)


def unsteady_time_domain(rho, u, b, dt, h, a, w=None):
    """Return Wagner-Küssner time-domain reference forces."""
    h, a = np.asarray(h, float) - np.mean(h), np.asarray(a, float) - np.mean(a)
    w = np.zeros_like(h) if w is None else np.asarray(w, float) - np.mean(w)
    time = np.arange(1, len(h)+1)*dt; tau, ds = time*u/b, dt*u/b
    hdot, adot = np.gradient(h, dt), np.gradient(a, dt)
    hp = np.gradient(hdot, dt)*b**2/u**2; ap, app = adot*b/u, np.gradient(adot, dt)*b**2/u**2
    phi_se = 1-.165*np.exp(-.089*tau)-.335*np.exp(-.6*tau); phi_b = 1-.5*np.exp(-.26*tau)-.5*np.exp(-2*tau)
    se_h, se_a, se_aa = _convolve(phi_se, hp/b, ds), _convolve(phi_se, ap, ds), _convolve(phi_se, app, ds/4)
    b_w = _convolve(np.gradient(phi_b, ds), w/u, ds)
    l_h = -np.pi*rho*u**2*hp/4 - .5*rho*u**2*b*2*np.pi*se_h; m_h = .5*rho*u**2*b**2*np.pi*se_h/2
    l_a = -np.pi*rho*u**2*b*ap/4 - .5*rho*u**2*b*2*np.pi*(se_a+se_aa); m_a = -np.pi*rho*b**2*u**2*(ap+app/8)/16 + .5*rho*u**2*b**2*np.pi*(se_a+se_aa)/2
    l_b, m_b = -.5*rho*u**2*b*2*np.pi*b_w, .5*rho*u**2*b**2*np.pi*b_w/2
    l = l_h + l_a + l_b; m = m_h + m_a + m_b
    l, m = l - l.mean(), m - m.mean()
    l_h, l_a, l_b = l_h - l_h.mean(), l_a - l_a.mean(), l_b - l_b.mean()
    m_h, m_a, m_b = m_h - m_h.mean(), m_a - m_a.mean(), m_b - m_b.mean()
    return l, m, time, l_h, l_a, l_b, m_h, m_a, m_b
