"""Compare analytical and learned flat-plate aerodynamic derivatives."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib.pyplot as plt
from self_excited_elm import flat_plate_derivatives, get_derivatives, harmonic_heave, harmonic_pitch
from common import COLORS, output_path, style, train_models

def main():
    style(); models, amp_h, amp_a, _ = train_models("linear"); rho,u,b,dt = 1.2,10.,31.,.01
    vr = np.linspace(1, 50, 30); analytical = flat_plate_derivatives(vr); predicted = []
    for value in vr:
        n = 3*int(np.ceil((b*value/u)/dt))
        _,_,_,h,hd,hdd = harmonic_heave(rho,u,b,value,amp_h,dt,n)
        _,_,_,a,ad,add = harmonic_pitch(rho,u,b,value,amp_a,dt,n)
        flh = models[0].predict(np.c_[h,hd,hdd])*.5*rho*u*u*b; fmh = models[1].predict(np.c_[h,hd,hdd])*.5*rho*u*u*b*b
        fla = models[2].predict(np.c_[a,ad,add])*.5*rho*u*u*b; fma = models[3].predict(np.c_[a,ad,add])*.5*rho*u*u*b*b
        predicted.append(get_derivatives(u,b,rho,value,flh,fmh,fla,fma,-h,-hd,a,ad))
    predicted = np.asarray(predicted)
    pairs = ((0, 3, r"$H_1^*, H_4^*$", -1, (-15, 5)),
             (1, 2, r"$H_2^*, H_3^*/2$", -1, (-25, 5)),
             (4, 7, r"$A_1^*, A_4^*$", 1, (0, 4)),
             (5, 6, r"$A_2^*, A_3^*/2$", 1, (-4, 6)))
    fig, axes = plt.subplots(2,2,figsize=(8,5.8),sharex=True,constrained_layout=True)
    for ax,(i,j,ylabel,sign,ylim) in zip(axes.flat,pairs):
        scale_j = .5 if j in (2,6) else 1
        ax.plot(vr, analytical[:,i], color=COLORS["blue"], label="Analytical first")
        ax.plot(vr, sign*predicted[:,i], "--", color=COLORS["blue"], label="ELM first")
        ax.plot(vr, analytical[:,j]*scale_j, color=COLORS["red"], label="Analytical second")
        ax.plot(vr, sign*predicted[:,j]*scale_j, "--", color=COLORS["red"], label="ELM second")
        ax.set(xlim=(2, 16), ylim=ylim, ylabel=ylabel); ax.legend(fontsize=8, ncol=2)
    for ax in axes[1]: ax.set_xlabel(r"Reduced velocity, $v_r$ [-]")
    fig.savefig(output_path("flat_plate_derivatives.pdf")); plt.close(fig)

if __name__ == "__main__": main()
