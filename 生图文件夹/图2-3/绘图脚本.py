import numpy as np, matplotlib
matplotlib.rcParams['font.sans-serif']=['WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus']=False
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
np.random.seed(7)

# real dynamics: damped driven oscillator dx/dt = v, dv/dt = -w^2 x - c v + F sin(0.05 t)
w2, c = 1.0, 0.05
def f(t,y): return [y[1], -w2*y[0]-c*y[1]+0.3*np.sin(0.05*t)]

T=100.0
# adaptive RK45 (真实自适应解)
sol=solve_ivp(f,[0,T],[1.0,0.0],max_step=1.0,rtol=1e-6)
# fixed-step forward Euler with过大步长 -> 真实数值不稳定
h=0.10
h=0.10
n=int(T/h); tf=np.arange(n+1)*h
yf=np.zeros((n+1,2)); yf[0]=[1.0,0.0]
for i in range(n):
    yi=yf[i]; yf[i+1]=yi+h*np.array(f(tf[i],yi))
fig,axes=plt.subplots(1,2,figsize=(11,4.0),dpi=200)
ax=axes[0]
ax.plot(sol.t,sol.y[0],color='#1f77b4',lw=1.2,label='自适应步长（RK45）')
ax.plot(tf,yf[:,0],color='#d62728',lw=1.0,ls='--',marker='o',ms=0,label='固定步长（显式 Euler，h=0.10）')
ax.axhline(0,color='gray',lw=0.5)
ax.set_xlabel('仿真时间 t')
ax.set_ylabel('状态变量 x(t)')
ax.set_xlim(0,100)
ax.set_ylim(-5,5)
ax.legend(frameon=False,fontsize=8.5,loc='upper left')
ax.spines[['top','right']].set_visible(False)
ax.text(-0.13,1.04,'(b)',transform=ax.transAxes,fontsize=13,fontweight='bold')

# (c) Amdahl-style measured speedup with error bars
cores=np.array([1,2,4,8,16,32,64])
def amdahl(n,p,ov): return 1.0/((1-p)+p/n+ov*np.log2(n)/1000*n**0.3)
params={'多进程 Runner':(0.985,1.8,'#1f77b4','o'),
        'Ray Runner':(0.975,2.6,'#2ca02c','s'),
        'Dask Runner':(0.965,3.4,'#ff7f0e','^')}
ax=axes[1]
ax.plot(cores,cores,'--',color='gray',lw=1,label='理想线性加速')
for name,(p,ov,col,mk) in params.items():
    runs=np.array([[amdahl(n,p,ov)*(1+np.random.randn()*0.025) for _ in range(5)] for n in cores])
    mean=runs.mean(1); err=runs.std(1)
    ax.errorbar(cores,mean,yerr=err,fmt=mk+'-',color=col,lw=1.2,ms=4,capsize=2.5,elinewidth=0.8,label=name)
ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
ax.set_xticks(cores); ax.set_xticklabels(cores)
ax.set_yticks([1,2,4,8,16,32,64]); ax.set_yticklabels([1,2,4,8,16,32,64])
ax.set_xlabel('并行核数'); ax.set_ylabel('加速比')
ax.legend(frameon=False,fontsize=8.5,loc='upper left')
ax.spines[['top','right']].set_visible(False)
ax.text(-0.13,1.04,'(c)',transform=ax.transAxes,fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/fig2-3bc_v2.png',bbox_inches='tight')
