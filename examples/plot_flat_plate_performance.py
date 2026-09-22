"""Compare analytical and learned forces for harmonic motion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib.pyplot as plt
from self_excited_elm import harmonic_heave, harmonic_pitch
from common import COLORS, output_path, style, train_models

def main():
    style(); models, amp_h, amp_a, _ = train_models("log", include_reduced_velocity=True); rho,u,b,dt,vr = 1.2,10.,31.,.01,5.
    n = int(np.ceil((b*vr/u)/dt))
    lh,mh,_,h,hd,hdd = harmonic_heave(rho,u,b,vr,amp_h,dt,n)
    la,ma,_,a,ad,add = harmonic_pitch(rho,u,b,vr,amp_a,dt,n)
    refs = ((h,lh/(.5*rho*u*u*b),mh/(.5*rho*u*u*b*b),models[0],models[1],r"$h/\hat h$ [-]",r"$C_{L,h}, C_{M,h}$ [-]"),(a,la/(.5*rho*u*u*b),ma/(.5*rho*u*u*b*b),models[2],models[3],r"$\alpha/\hat\alpha$ [-]",r"$C_{L,\alpha}, C_{M,\alpha}$ [-]"))
    fig, axes = plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    for ax,(q,cl,cm,model_l,model_m,xlab,ylab) in zip(axes,refs):
        x=np.c_[q, np.gradient(q,dt), np.gradient(np.gradient(q,dt),dt), np.full(len(q), vr)]; qn=q/np.max(np.abs(q))
        ax.plot(qn,cl,color=COLORS["blue"],alpha=.6,label="Analytical lift"); ax.plot(qn,model_l.predict(x),"--",color=COLORS["blue"],label="ELM lift")
        ax.plot(qn,cm,color=COLORS["red"],alpha=.6,label="Analytical moment"); ax.plot(qn,model_m.predict(x),"--",color=COLORS["red"],label="ELM moment")
        ax.set(xlabel=xlab,ylabel=ylab); ax.legend(fontsize=8,ncol=2)
    fig.savefig(output_path("flat_plate_performance.pdf")); plt.close(fig)

if __name__ == "__main__": main()
