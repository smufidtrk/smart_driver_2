"""
Weighted-centroid localisation (WCL) on the 32-box captures.

  python analysis/phase_slope.py       (first: writes sessions/phase_slope/phase_slope.csv)
  python analysis/weighted_centroid.py

x_est = sum(w_i * x_antenna_i) / sum(w_i) with weights from RSSI (mW), the relative phase
slope (exp), or both.  Zone by the fixed 12" edges ("raw") or after a leave-one-out linear map
x_est -> x ("cal").  Compared with simply taking the antenna with the largest weight (argmax).
Writes sessions/phase_slope/wcl_per_box.csv.
"""
import numpy as np, pandas as pd
R=pd.read_csv("sessions/phase_slope/phase_slope.csv")
XA={1:6,2:18,3:30,4:42}; EDGES=[12,24,36]
zone_of=lambda x: np.searchsorted(EDGES, x, side="right")+1
rows=[]; per=[]
for cap,G in R.groupby("capture",sort=False):
    G=G.copy()
    G["rel_sl"]=(G.slope-G.groupby("ant").slope.transform("median"))/(G.slope-G.groupby("ant").slope.transform("median")).std()
    G["rel_rs"]=G.rssi-G.groupby("ant").rssi.transform("median")
    W={}
    W["RSSI mW"]=lambda g: 10**(g.rssi/10)
    W["RSSI mW^2"]=lambda g: 10**(g.rssi/5)
    W["RSSI mW, antenna-normalised"]=lambda g: 10**(g.rel_rs/10)
    W["slope exp(rel)"]=lambda g: np.exp(g.rel_sl)
    W["slope exp(2 rel)"]=lambda g: np.exp(2*g.rel_sl)
    W["slope + RSSI"]=lambda g: np.exp(g.rel_sl)*10**(g.rel_rs/10)
    for name,f in W.items():
        out=[]
        for box,g in G.groupby("box"):
            g=g.dropna(subset=["slope","rssi"]); 
            if g.empty: continue
            w=f(g).values; xh=(w*g.ant.map(XA).values).sum()/w.sum()
            amax=g.ant.values[np.argmax(w)]
            out.append(dict(box=box,x=g.x.iloc[0],zone=g.zone.iloc[0],xh=xh,amax=amax))
        O=pd.DataFrame(out)
        # calibrated: leave-one-out linear map xh -> x
        cal=[]
        for i in range(len(O)):
            o=O.drop(index=i); k=np.polyfit(o.xh,o.x,1); cal.append(np.polyval(k,O.xh.iloc[i]))
        O["xc"]=cal
        per.append(O.assign(capture=cap,method=name))
        rows.append(dict(capture=cap,method=name,
            corr=O.xh.corr(O.x), spread=O.xh.std()/O.x.std(),
            mae_raw=(O.xh-O.x).abs().mean(), mae_cal=(O.xc-O.x).abs().mean(),
            zone_raw=(zone_of(O.xh)==O.zone).mean(), zone_cal=(zone_of(O.xc)==O.zone).mean(),
            zone_argmax=(O.amax==O.zone).mean(),
            within1=(np.abs(zone_of(O.xc)-O.zone)<=1).mean()))
D=pd.DataFrame(rows)
S=D.groupby("method",sort=False)[["corr","spread","mae_raw","mae_cal","zone_raw","zone_cal","zone_argmax","within1"]].mean().round(2)
pd.set_option("display.width",200); print(S.to_string())
P=pd.concat(per); P.to_csv("sessions/phase_slope/wcl_per_box.csv",index=False)
# error by true x (edge effect) for slope+RSSI calibrated
q=P[P.method=="slope + RSSI"]
q = P[P.method == "slope + RSSI"].reset_index(drop=True)
q["err"] = (q.xc - q.x).abs()
q["ok"] = zone_of(q.xc) == q.zone
print("\nslope + RSSI, calibrated: estimate and |error| (in) by true x")
print(q.groupby("x").agg(estimate=("xc", "mean"), error=("err", "mean"), zone_ok=("ok", "mean")).round(1).T.to_string())
