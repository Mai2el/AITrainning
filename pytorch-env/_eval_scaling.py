"""Đánh giá chất lượng scale per-feature: áp đúng pipeline, đo token ×30000."""
import numpy as np, pandas as pd
import tabular_tokenizer as T
from sklearn.model_selection import train_test_split

SF = T.DEFAULT_SCALE_FACTOR
df = pd.read_csv("BCCC_selected_features_raw.csv", low_memory=False)
df.columns = df.columns.str.strip()
df.drop(columns=[c for c in ["label","flow_id","timestamp","src_ip","dst_ip",
        "src_port","active_std","active_variance"] if c in df.columns], inplace=True)
for col in df.columns:
    if col != "activity":
        df[col] = pd.to_numeric(df[col], errors="coerce")
df.replace([np.inf,-np.inf], np.nan, inplace=True); df.fillna(0, inplace=True)
y = pd.factorize(df["activity"])[0]
features = [c for c in df.columns if c != "activity"]
idx_tr,_ = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=42)

_FIXED = ["std_fwd_header_bytes_delta_len","std_bwd_packets_delta_time","fwd_max_header_bytes",
    "fwd_median_header_bytes","fwd_min_header_bytes","fwd_mode_header_bytes","max_header_bytes",
    "median_header_bytes","min_header_bytes","mode_header_bytes","delta_start",
    "syn_flag_percentage_in_total","mean_header_bytes","fwd_init_win_bytes","bwd_max_header_bytes",
    "bwd_ece_flag_percentage_in_total","bwd_ece_flag_counts","bwd_packets_IAT_median",
    "bwd_payload_bytes_max","bwd_psh_flag_percentage_in_bwd_packets","ece_flag_percentage_in_total",
    "fwd_cov_header_bytes","fwd_ece_flag_counts","fwd_psh_flag_percentage_in_fwd_packets"]
fixed = [c for c in _FIXED if c in features]
print(f"\n[fixed_normal có trong dataset: {len(fixed)}/{len(_FIXED)}]")
profiles = T.phase1_profile(df, features, idx_tr)
groups, STRAT = T.phase2_group(profiles, fixed_normal_features=fixed,
    report_path="_eval_groups.json", scale_factor=SF, df=df, idx_train=idx_tr)

cont = []
for grp in (T.GROUP_NORMAL, T.GROUP_SKEWED):
    cont += [(c,grp) for c in groups[grp]]

print("\n%-40s %-7s %-13s %5s %6s %7s %6s %6s %5s" % (
    "feature","grp","strategy","nIn","nTok","keep%","bulk","tmax%","clip"))
print("-"*110)
rows=[]
for col,grp in sorted(cont):
    p = profiles[col]
    xt = df.loc[idx_tr, col].values.astype(float).reshape(-1,1)
    strat = STRAT[col]
    clipped = not T._fits_token_budget(p, SF)
    xc,_ = (T._apply_universal_clip(xt, xt, [p], SF))
    if strat==T.STRAT_CBRT: tok=T._signed_power_scale(xc,xc,SF,1/3)
    elif strat==T.STRAT_SQRT: tok=T._signed_power_scale(xc,xc,SF,1/2)
    elif strat==T.STRAT_QUANTILE: tok=T._quantile_scale(xc,xc,SF)
    else:
        lp,hp = (1,99) if (clipped) else (0,100)
        tok=T._robust_linear_scale(xc,xc,SF,lo_pct=lp,hi_pct=hp)
    tok=tok.flatten()
    nIn=int(np.unique(xt).size); nTok=int(np.unique(tok).size)
    keep = nTok/min(nIn,SF+1)*100
    # bulk: số token phân biệt trong khoảng p5..p95 của token
    lo,hi=np.percentile(tok,5),np.percentile(tok,95)
    bulk=int(np.unique(tok[(tok>=lo)&(tok<=hi)]).size)
    tmax=tok.max()/SF*100
    rows.append((col,grp,strat,nIn,nTok,keep,bulk,tmax,clipped))
    print("%-40s %-7s %-13s %5d %6d %6.1f %6d %5.0f %5s"%(
        col,grp[:7],strat,nIn,nTok,keep,bulk,tmax,clipped))

print("\n=== CỜ NGHI NGỜ CHƯA TỐI ƯU ===")
for col,grp,strat,nIn,nTok,keep,bulk,tmax,clipped in rows:
    flags=[]
    if nIn>50 and keep<50: flags.append("MẤT>50%% giá trị (keep=%.0f%%)"%keep)
    if nIn>500 and bulk<30: flags.append("THÂN bị nén (bulk=%d token)"%bulk)
    if tmax<60 and nIn>20: flags.append("LÃNG PHÍ dải (tmax=%.0f%%)"%tmax)
    if flags: print("  %-40s [%s] -> %s"%(col,strat,"; ".join(flags)))
