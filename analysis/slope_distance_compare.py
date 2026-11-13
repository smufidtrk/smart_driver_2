"""
Slope score: "relative to each antenna's median" (used by the Zone Locator) vs a per-antenna
distance fit (slope = c + k*|x - antenna x|, fitted on the lookup boxes; the tag's x is the
position that best explains its slopes on all antennas; locator.slope_model / slope_position).
Slope only, no phase or RSSI.  Same test sets as the locator tuning.

  python analysis/slope_distance_compare.py
"""
import sys, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "zone_ui")); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import locator as L
T={}
for pw in (15.0,20.0):
    cal=L.library(["4zone_cal32"],pw); rep=L.library(["4zone_cal32_repeat"],pw); both=cal+rep
    T[f"on: minutes @{pw:g}"]=[(t,both) for t in L.library(["4zone_mode2_d"],pw)]
    T[f"on: overnight @{pw:g}"]=[(t,cal) for t in rep]
    T[f"off: 6in LOO @{pw:g}"]=[(t,[c for c in cal if c is not t]) for t in cal]
    tb=L.library(["4zone_12box"],pw)
    if tb: T[f"off: 12box @{pw:g}"]=[(t,both) for t in tb]
zones=sorted(L.PZ.values())
def old(t,lib):
    med,sd=L.slope_reference(lib); r=(L._slopes(t)-med)/sd
    z=np.array([r[L.PORTS.index(next(p for p in L.PORTS if L.PZ[p]==zz))] for zz in zones])
    return zones[int(np.nanargmax(z))] if np.isfinite(z).sum()>=2 else None
models={}
def new(t,lib,nu,scale):
    key=id(lib) if len(lib)>40 else None
    m=models.get(key) if key else None
    if m is None:
        m=L.slope_model(lib); 
        if key: models[key]=m
    L.SLOPE_NU, L.SLOPE_SCALE = nu, scale
    r=L.slope_position(t,m) if m else None
    return (zones[int(np.argmax(r["p"]))], r["x"]) if r else (None,None)
rows = []
cfgs={"old: relative to median":None,"distance fit, Gaussian":(1e6,1.0),"distance fit, robust t4":(4.0,1.0),"distance fit, robust t2":(2.0,1.0)}
err=[]
for name,c in cfgs.items():
    r={"method":name}; on=off=0
    for k,v in T.items():
        ok=0
        for t,lib in v:
            if c is None: z=old(t,lib)
            else:
                z,x=new(t,lib,*c)
                if name=="distance fit, robust t4": err.append(dict(test=k,x=t["x"],xh=x))
            ok+= z==t["zone"]
        r[k]=f"{ok}/{len(v)}"
        if k.startswith("on"): on+=ok
        else: off+=ok
    r["on /128"]=on; r["off /76"]=off; rows.append(r)
pd.set_option("display.width",250); print(pd.DataFrame(rows).to_string(index=False))
E=pd.DataFrame(err); E=E[~E.test.str.contains("12box")]
print("\nrobust t4 x estimate: mean |error| %.1f in, corr with true x %.2f" % ((E.xh-E.x).abs().mean(), E.xh.corr(E.x)))
m=L.slope_model(L.library(["4zone_cal32","4zone_cal32_repeat"],20.0))
for p,f in m.items(): print(f"A{p} @20: slope = {f['c']:.1f} {f['k']:+.2f}*|dx|  scatter {f['sd']:.1f} deg/MHz = {f['sd_in']:.0f} in")
