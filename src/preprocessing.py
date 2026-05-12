import os
import random
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype

warnings.filterwarnings("ignore")

SEED = 1
MODE = "STRICT_CAUSAL"   # or "NON_CAUSAL"
DATA_DIR = "../data" # Relative to src/ or run directory

PATH_MESSAGES  = os.path.join(DATA_DIR, "messages-final-balancing.csv")
PATH_CAMPAIGNS = os.path.join(DATA_DIR, "campaigns.csv")
PATH_CLIENTS   = os.path.join(DATA_DIR, "client_first_purchase_date.csv")
PATH_HOLIDAYS  = os.path.join(DATA_DIR, "holidays.csv")

TZ_LOCAL = "Asia/Seoul"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
rng = np.random.RandomState(SEED)

to_dt = lambda s: pd.to_datetime(s, errors="coerce", utc=True)

# =====================================================
# 1) BASIC UTILS
# =====================================================
def drop_if_exists(df, cols):
    return df.drop([c for c in cols if c in df.columns], axis=1, errors="ignore")

def reduce_email_provider(ep_series):
    ep = ep_series.astype(str).str.lower().str.strip()
    bad = {"", "nan", "none", "null"}
    ep = ep.where(~ep.isin(bad), "other")
    forced = ["gmail.com", "mail.ru"]
    top3 = forced
    ep_reduced = np.where(ep.isin(top3), ep, "other")
    return ep_reduced, top3

def _norm_str(x):
    if pd.isna(x): return None
    return str(x).strip().lower().replace(" ", ".")

def onehot_join(df, col, prefix):
    if col not in df.columns:
        return pd.DataFrame(index=df.index)
    vals = df[col].copy()
    mask = vals.notna()
    labels = prefix + vals[mask].astype(str)
    oh = pd.get_dummies(labels, dtype=int)
    oh = oh.reindex(index=df.index, fill_value=0)
    return oh

def hours_since(now, past):
    now  = pd.to_datetime(now,  errors="coerce", utc=True)
    past = pd.to_datetime(past, errors="coerce", utc=True)
    return (now - past).dt.total_seconds() / 3600.0

def cum_nunique(series):
    seen=set(); out=[]
    for v in series:
        if v not in seen: seen.add(v)
        out.append(len(seen))
    return pd.Series(out, index=series.index)

def cum_nunique_hist(s):
    seen=set(); out=[]
    for v in s:
        out.append(len(seen))
        seen.add(v)
    return pd.Series(out, index=s.index)

# =====================================================
# 2) LOAD + CLEAN + MERGE
# =====================================================
def load_and_preprocess(
    path_messages=PATH_MESSAGES,
    path_campaigns=PATH_CAMPAIGNS,
    path_clients=PATH_CLIENTS,
    path_holidays=PATH_HOLIDAYS,
    mode=MODE,
    tz_local=TZ_LOCAL,
):
    # ---------- Load ----------
    messages  = pd.read_csv(path_messages)
    campaigns = pd.read_csv(path_campaigns)
    clients   = pd.read_csv(path_clients)
    holidays  = pd.read_csv(path_holidays)

    # (Raw campaigns copy for ECC toggles/ab_test)
    try:
        campaigns_raw_for_toggles = pd.read_csv(path_campaigns)
    except Exception:
        campaigns_raw_for_toggles = None

    # ---------- Drop ----------
    campaigns = drop_if_exists(campaigns, [
        "is_test","position","warmup_mode","hour_limit",
        "subject_length","total_count"
    ])
    messages = drop_if_exists(messages, [
        "category","id","created_at","updated_at","stream",
    ])

    # ---------- campaigns: f/t -> 0/1, datetime ----------
    for col in campaigns.columns:
        if campaigns[col].dtype == object and campaigns[col].dropna().isin(["f","t"]).all():
            campaigns[col] = campaigns[col].map({"t":1,"f":0}).astype("Int64")
    for c in ["started_at","finished_at"]:
        if c in campaigns.columns:
            campaigns[c] = to_dt(campaigns[c])

    # ---------- clients ----------
    if "first_purchase_date" in clients.columns:
        clients["first_purchase_date"] = to_dt(clients["first_purchase_date"])

    # ---------- holidays ----------
    holidays["date"] = to_dt(holidays["date"])
    if "is_holidays" in holidays.columns and "is_holiday" not in holidays.columns:
        holidays = holidays.rename(columns={"is_holidays":"is_holiday"})
    if "is_holiday" not in holidays.columns:
        holidays["is_holiday"] = 0

    # ---------- messages flags & datetimes ----------
    flag_cols = [
        "is_opened","is_unsubscribed","is_complained","is_purchased","is_clicked",
        "is_hard_bounced","is_soft_bounced","is_blocked"
    ]
    for c in flag_cols:
        if c in messages.columns:
            messages[c] = messages[c].map({"t":1,"f":0}).fillna(0).astype(int)

    dt_cols = [
        "sent_at","opened_first_time_at","opened_last_time_at",
        "clicked_first_time_at","clicked_last_time_at",
        "complained_at","purchased_at","unsubscribed_at","date",
        "hard_bounced_at","soft_bounced_at","blocked_at"
    ]
    for c in dt_cols:
        if c in messages.columns:
            messages[c] = to_dt(messages[c])

    # ---------- campaigns standardization + one-hot ----------
    for col in ["campaign_type","channel","topic"]:
        if col in campaigns.columns:
            campaigns[col] = campaigns[col].apply(_norm_str)

    rename_map = {}
    if "campaign_type" in campaigns.columns and "camp_campaign_type" not in campaigns.columns:
        rename_map["campaign_type"] = "camp_campaign_type"
    if "channel" in campaigns.columns and "camp_channel" not in campaigns.columns:
        rename_map["channel"] = "camp_channel"
    if "topic" in campaigns.columns and "camp_topic" not in campaigns.columns:
        rename_map["topic"] = "camp_topic"
    if rename_map:
        campaigns = campaigns.rename(columns=rename_map)

    ALLOWED_TYPE  = {"bulk", "transactional", "trigger"}
    ALLOWED_CH    = {"email", "mobile_push", "multichannel", "sms"}
    ALLOWED_TOPIC = {"event", "happy.birthday", "leave.review", "offer.after.purchase", "sale.out"}

    if "camp_campaign_type" in campaigns.columns:
        campaigns["camp_campaign_type"] = np.where(
            campaigns["camp_campaign_type"].isin(ALLOWED_TYPE),
            campaigns["camp_campaign_type"], None
        )
    if "camp_channel" in campaigns.columns:
        campaigns["camp_channel"] = np.where(
            campaigns["camp_channel"].isin(ALLOWED_CH),
            campaigns["camp_channel"], None
        )
    if "camp_topic" in campaigns.columns:
        def _topic_map(v):
            if v is None: return None
            return v if v in ALLOWED_TOPIC else "other"
        campaigns["camp_topic"] = campaigns["camp_topic"].apply(_topic_map)

    oh_type  = onehot_join(campaigns, "camp_campaign_type", "camp_campaign_type")
    oh_chan  = onehot_join(campaigns, "camp_channel",       "camp_channel")
    oh_topic = onehot_join(campaigns, "camp_topic",         "camp_topic")

    if "id" in campaigns.columns:
        camp_base = campaigns[["id","started_at","finished_at"]].rename(columns={"id":"campaign_id"})
    else:
        camp_base = campaigns[["campaign_id","started_at","finished_at"]]

    campaigns_oh = pd.concat([camp_base, oh_type, oh_chan, oh_topic], axis=1)

    # ---------- Merge ----------
    merged = messages.merge(
        clients[["client_id","first_purchase_date"]], on="client_id", how="left"
    )

    keep_camp = ["campaign_id","started_at","finished_at"] + \
                [c for c in campaigns_oh.columns if c.startswith(("camp_campaign_type","camp_channel","camp_topic"))]
    keep_camp = [c for c in keep_camp if c in campaigns_oh.columns]
    merged = merged.merge(campaigns_oh[keep_camp], on="campaign_id", how="left")

    # Holiday merge (thesis-like)
    holidays_0 = holidays.copy(); holidays_0["is_purchased"] = 0
    holidays_1 = holidays.copy(); holidays_1["is_purchased"] = 1
    temp_holidays = pd.concat([holidays_0, holidays_1], ignore_index=True)

    final_data = merged.merge(
        temp_holidays[["date","is_holiday","is_purchased"]].drop_duplicates(),
        on=["date","is_purchased"],
        how="left"
    )

    final_data = final_data.sort_values(["client_id","sent_at"]).reset_index(drop=True)
    final_data["campaign_id"] = final_data["campaign_id"].fillna("NA")
    if "is_purchased" not in final_data.columns:
        final_data["is_purchased"] = 0

    # Preserve raw channel/provider
    if "channel" in final_data.columns:
        final_data["_channel_raw"] = final_data["channel"].astype(str).str.lower().str.strip()
    else:
        final_data["_channel_raw"] = "unknown"

    if "email_provider" in final_data.columns:
        typo_map = {
            "gmail.com.com":"gmail.com","gmajl.com":"gmail.com","gmaol.com":"gmail.com",
            "icoud.com":"icloud.com","yandek.ru":"yandex.ru","yangex.ru":"yandex.ru",
        }
        final_data["email_provider"] = (
            final_data["email_provider"].astype(str).str.lower().str.strip().replace(typo_map)
        )
        final_data["email_provider_raw"] = final_data["email_provider"]
        ep_reduced, EP_TOP3 = reduce_email_provider(final_data["email_provider"])
        final_data["email_provider"] = ep_reduced
    else:
        EP_TOP3 = []
        final_data["email_provider_raw"] = "other"

    # one-hot for certain cols
    onehot_cols = [c for c in ["channel","message_type","email_provider"] if c in final_data.columns]
    final_data = pd.get_dummies(final_data, columns=onehot_cols, dtype=int)

    # platform normalization -> platform.desktop/smartphone/... + platform.(other)
    if "platform" in final_data.columns:
        p = final_data["platform"].astype(str).str.lower().str.strip()
        def map_platform(v):
            if v in ("nan", "", "none"): return "other"
            if v == "desktop":    return "desktop"
            if v == "smartphone": return "smartphone"
            if v == "phablet":    return "phablet"
            if v == "tablet":     return "tablet"
            return "other"
        plat = p.map(map_platform)
        final_data = final_data.drop(columns=["platform"])
        fixed = ["desktop","smartphone","phablet","tablet"]
        for name in fixed:
            final_data[f"platform.{name}"] = (plat == name).astype(int)
        final_data["platform."] = (plat == "other").astype(int)

    # Merge ECC toggles (+ ab_test) from raw campaigns
    ecc_toggle_cols = []
    if campaigns_raw_for_toggles is not None:
        cand_cols = [c for c in campaigns_raw_for_toggles.columns if str(c).startswith("subject_with_")]
        for col in cand_cols:
            s = campaigns_raw_for_toggles[col]
            if s.dtype == object and s.dropna().isin(["t","f"]).all():
                campaigns_raw_for_toggles[col] = s.map({"t":1,"f":0}).astype("Int64")
        merge_cols = ["id"] + cand_cols
        if "ab_test" in campaigns_raw_for_toggles.columns:
            merge_cols += ["ab_test"]
        togg_df = campaigns_raw_for_toggles[merge_cols].rename(columns={"id":"campaign_id"})
        final_data = final_data.merge(togg_df, on="campaign_id", how="left")
        ecc_toggle_cols = cand_cols

    # ---------- STRICT_CAUSAL baseline aggregates ----------
    RECENCY_MAP = [
        ("first_purchase_date",  "avg_time_since_first_purchase"),
        ("opened_last_time_at",  "avg_time_since_last_open"),
        ("clicked_last_time_at", "avg_time_since_last_click"),
        ("unsubscribed_at",      "avg_time_since_unsubscribe"),
        ("complained_at",        "avg_time_since_complaint"),
    ]

    if mode == "STRICT_CAUSAL":
        final_data["total_messages"] = final_data.groupby("client_id").cumcount()
        final_data["total_campaigns"] = (
            final_data.groupby("client_id")["campaign_id"]
                      .apply(cum_nunique_hist).reset_index(level=0, drop=True).astype(np.int32)
        )
        final_data["total_purchases"] = (
            final_data.groupby("client_id")["is_purchased"].cumsum().shift(1).fillna(0).astype(np.int32)
        )
        for src, new in RECENCY_MAP:
            if src in final_data.columns:
                prior_known = final_data.groupby("client_id")[src].ffill().shift(1)
                rec = hours_since(final_data["sent_at"], prior_known)
                med = pd.to_numeric(rec, errors="coerce").replace([np.inf,-np.inf],np.nan).median(skipna=True)
                med = float(med) if np.isfinite(med) else 0.0
                final_data[new] = (pd.to_numeric(rec, errors="coerce")
                                   .replace([np.inf,-np.inf],np.nan).fillna(med)
                                   .clip(lower=0).astype(np.float32))
        if {"finished_at","started_at"}.issubset(final_data.columns):
            final_data["campaign_duration"] = (
                (final_data["finished_at"] - final_data["started_at"]).dt.total_seconds()/3600.0
            ).astype("float32")
            final_data["avg_campaign_duration"] = final_data.groupby("client_id")["campaign_duration"].shift(1)
            med = pd.to_numeric(final_data["avg_campaign_duration"], errors="coerce") \
                    .replace([np.inf,-np.inf],np.nan).median(skipna=True)
            med = float(med) if np.isfinite(med) else 0.0
            final_data["avg_campaign_duration"] = (pd.to_numeric(final_data["avg_campaign_duration"], errors="coerce")
                                                  .fillna(med).clip(lower=0).astype(np.float32))
    else:
        final_data["total_messages"] = final_data.groupby("client_id").cumcount() + 1
        final_data["total_campaigns"] = (
            final_data.groupby("client_id")["campaign_id"]
                      .apply(cum_nunique).reset_index(level=0, drop=True).astype(np.int32)
        )
        final_data["total_purchases"] = final_data.groupby("client_id")["is_purchased"].cumsum().astype(np.int32)
        for src, new in RECENCY_MAP:
            if src in final_data.columns:
                prior_known = final_data.groupby("client_id")[src].ffill()
                rec = hours_since(final_data["sent_at"], prior_known)
                med = pd.to_numeric(rec, errors="coerce").replace([np.inf,-np.inf],np.nan).median(skipna=True)
                med = float(med) if np.isfinite(med) else 0.0
                final_data[new] = (pd.to_numeric(rec, errors="coerce")
                                   .replace([np.inf,-np.inf],np.nan).fillna(med)
                                   .clip(lower=0).astype(np.float32))
        if {"finished_at","started_at"}.issubset(final_data.columns):
            final_data["campaign_duration"] = (
                (final_data["finished_at"] - final_data["started_at"]).dt.total_seconds()/3600.0
            ).astype("float32")
            final_data["avg_campaign_duration"] = (
                final_data.groupby("client_id")["campaign_duration"]
                .expanding().mean().reset_index(level=0, drop=True).astype("float32")
            )

    return final_data, ecc_toggle_cols, EP_TOP3

# =====================================================
# 3) FEATURE ENGINEERING FUNCTIONS
# =====================================================
def add_ready_to_buy_hazard(df):
    df = df.sort_values(["client_id","sent_at"]).copy()
    pur_col = df["purchased_at"].where(df["is_purchased"]==1) if "purchased_at" in df.columns else df["sent_at"].where(df["is_purchased"]==1)
    df["_last_purchase_time"] = pur_col.groupby(df["client_id"]).ffill().shift(1)

    d_days = (df["sent_at"] - df["_last_purchase_time"]).dt.total_seconds()/86400.0
    med_dd = d_days.median(skipna=True)
    med_dd = float(med_dd) if np.isfinite(med_dd) else 0.0
    df["days_since_last_purchase"] = d_days.fillna(med_dd)

    pur_times = df.loc[df["is_purchased"]==1, ["client_id", "_last_purchase_time", "sent_at"]].dropna()
    pur_times["gap_days"] = (pur_times["sent_at"] - pur_times["_last_purchase_time"]).dt.total_seconds()/86400.0
    stats = pur_times.groupby("client_id")["gap_days"].agg(["median","std"]).rename(columns={"median":"mu","std":"sigma"})

    global_mu = float(pur_times["gap_days"].median()) if len(pur_times) else 14.0
    global_sigma = float(pur_times["gap_days"].std()) if len(pur_times) and np.isfinite(pur_times["gap_days"].std()) else max(2.0, global_mu/3.0)
    stats = stats.replace([np.inf,-np.inf], np.nan).fillna({"mu":global_mu, "sigma":global_sigma})

    df = df.merge(stats, on="client_id", how="left").fillna({"mu":global_mu, "sigma":global_sigma})
    z = (df["days_since_last_purchase"] - df["mu"]) / (df["sigma"] + 1e-6)
    df["feat_rtb_hazard"] = np.exp(-0.5 * (z**2)).astype(np.float32)
    return df.drop(columns=["_last_purchase_time","mu","sigma"], errors="ignore")

def add_hour_shift_delta(df):
    df = df.sort_values(["client_id","sent_at"]).copy()
    resp_time = df["opened_first_time_at"].copy()
    if "clicked_first_time_at" in df.columns: resp_time = resp_time.fillna(df["clicked_first_time_at"])
    if "purchased_at" in df.columns:          resp_time = resp_time.fillna(df["purchased_at"])

    angle = (resp_time.dt.hour.fillna(0) + resp_time.dt.minute.fillna(0)/60.0) * 2*np.pi/24.0
    sin_a = np.sin(angle).fillna(0.0); cos_a = np.cos(angle).fillna(0.0)
    csin = sin_a.groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)
    ccos = cos_a.groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)
    n    = df["is_opened"].groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)

    global_angle = ((df["sent_at"].dt.hour.mean() or 12.0)) * 2*np.pi/24.0
    pref_angle = np.where(n>0, np.arctan2(csin, ccos), global_angle)

    sent_angle = (df["sent_at"].dt.hour + df["sent_at"].dt.minute/60.0) * 2*np.pi/24.0
    delta = np.abs(np.arctan2(np.sin(sent_angle - pref_angle), np.cos(sent_angle - pref_angle)))
    df["feat_hour_shift"] = (2.0*np.sin(delta/2.0)).astype(np.float32)
    return df

def add_dow_shift_delta(df):
    d = df.sort_values(["client_id","sent_at"]).copy()
    resp_ts = d["opened_first_time_at"]
    if "clicked_first_time_at" in d.columns: resp_ts = resp_ts.fillna(d["clicked_first_time_at"])
    if "purchased_at" in d.columns:          resp_ts = resp_ts.fillna(d["purchased_at"])

    resp_dow = resp_ts.dt.dayofweek
    ang = (2*np.pi/7.0) * resp_dow
    sin_a = np.sin(ang).fillna(0.0); cos_a = np.cos(ang).fillna(0.0)

    csin = sin_a.groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)
    ccos = cos_a.groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)
    n    = d["is_opened"].groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)

    global_ang = (2*np.pi/7.0) * d["sent_at"].dt.dayofweek.mean()
    pref = np.where(n>0, np.arctan2(csin, ccos), global_ang)

    now_ang = (2*np.pi/7.0) * d["sent_at"].dt.dayofweek
    delta = np.abs(np.arctan2(np.sin(now_ang - pref), np.cos(now_ang - pref)))
    d["feat_dow_shift"] = (2.0*np.sin(delta/2.0)).astype(np.float32)
    return d

def add_fatigue_and_cooldown(df, tau_map=None, cooldown_map=None):
    df = df.sort_values(["client_id","sent_at"]).copy()
    if tau_map is None:
        tau_map = {"email":48.0, "mobile_push":24.0, "sms":72.0, "multichannel":48.0, "unknown":48.0}
    if cooldown_map is None:
        cooldown_map = {"email":24.0, "mobile_push":6.0, "sms":24.0, "multichannel":24.0, "unknown":24.0}

    def per_client(g):
        last_time = {}
        out_fatigue, out_cool_ok = [], []
        for t, ch in zip(g["sent_at"], g["_channel_raw"]):
            ch = ch if ch in tau_map else "unknown"
            sc = 0.0
            for cc, tt in last_time.items():
                dt_h = (t - tt).total_seconds()/3600.0
                if dt_h >= 0: sc += np.exp(-dt_h / tau_map.get(cc, 48.0))
            out_fatigue.append(sc)

            last = last_time.get(ch, None)
            if last is None:
                ok = 1
            else:
                dt_h = (t - last).total_seconds()/3600.0
                ok = 1 if dt_h >= cooldown_map.get(ch, 24.0) else 0
            out_cool_ok.append(ok)

            last_time[ch] = t

        g["feat_fatigue"] = np.array(out_fatigue, dtype=np.float32)
        g["feat_cooldown_ok"] = np.array(out_cool_ok, dtype=np.int32)
        return g

    return df.groupby("client_id", group_keys=False).apply(per_client)

def add_ecc_distance(df, toggles):
    if not toggles:
        df["feat_ecc_hamming"] = 0.0
        return df

    df = df.sort_values(["client_id","sent_at"]).copy()
    for col in toggles:
        if col in df.columns: df[col] = df[col].fillna(0).astype(int)
        else:                 df[col] = 0

    a, b, theta = 1.0, 1.0, 0.0
    for col in toggles:
        on_flag  = df[col].astype(int)
        off_flag = 1 - on_flag
        c_on  = (df["is_purchased"]*on_flag).groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)
        n_on  = (on_flag).groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)
        c_off = (df["is_purchased"]*off_flag).groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)
        n_off = (off_flag).groupby(df["client_id"]).cumsum().shift(1).fillna(0.0)
        th_on, th_off = (c_on + a)/(n_on + a + b), (c_off + a)/(n_off + a + b)
        df[f"_pref_{col}"] = (th_on - th_off > theta).astype(int)

    pref_cols = [f"_pref_{c}" for c in toggles]
    pref_mat  = df[pref_cols].to_numpy(dtype=np.int32)
    now_mat   = df[toggles].to_numpy(dtype=np.int32)
    xor_mat   = np.abs(pref_mat - now_mat)
    df["feat_ecc_hamming"] = (xor_mat.sum(axis=1) / max(1, len(toggles))).astype(np.float32)
    return df.drop(columns=pref_cols, errors="ignore")

def add_microclimate_z_v2(df, window_days=56, bin_hours=2, clip_val=5.0):
    d = df.sort_values("sent_at").copy()
    sent_local = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)
    d["sent_at_local"] = sent_local
    d["mc_dow"]       = sent_local.dt.dayofweek.astype("Int16")
    d["mc_hour_bin"]  = ((sent_local.dt.hour // bin_hours) * bin_hours).astype("Int16")

    prov = d.get("email_provider_raw", "other")
    if isinstance(prov, str): prov = pd.Series([prov]*len(d), index=d.index)
    d["mc_prov"] = prov.fillna("other").astype(str)

    d["_is_opened"] = pd.to_numeric(d.get("is_opened", 0), errors="coerce").fillna(0).astype(int)

    d["mc_cell"]     = d["mc_prov"] + "|" + d["mc_dow"].astype(str) + "|" + d["mc_hour_bin"].astype(str)
    d["mc_prov_dow"] = d["mc_prov"] + "|" + d["mc_dow"].astype(str)

    mc_cell_sum = np.zeros(len(d), dtype=np.float32)
    mc_cell_cnt = np.zeros(len(d), dtype=np.float32)

    for _, g in d.groupby("mc_cell", sort=False):
        g2 = g.loc[g["sent_at_local"].notna()].sort_values("sent_at_local")
        if g2.empty: continue
        idx2 = g2.index; ts = g2["sent_at_local"]
        open_s = pd.Series(g2["_is_opened"].astype(float).values, index=ts)
        ones   = pd.Series(1.0, index=ts)
        rs = open_s.rolling(f"{window_days}D", closed="left").sum()
        rc = ones.rolling(f"{window_days}D", closed="left").sum()
        mc_cell_sum[idx2] = rs.values
        mc_cell_cnt[idx2] = rc.values

    mc_prov_dow_sum = np.zeros(len(d), dtype=np.float32)
    mc_prov_dow_cnt = np.zeros(len(d), dtype=np.float32)

    for _, g in d.groupby("mc_prov_dow", sort=False):
        g2 = g.loc[g["sent_at_local"].notna()].sort_values("sent_at_local")
        if g2.empty: continue
        idx2 = g2.index; ts = g2["sent_at_local"]
        open_s = pd.Series(g2["_is_opened"].astype(float).values, index=ts)
        ones   = pd.Series(1.0, index=ts)
        rs = open_s.rolling(f"{window_days}D", closed="left").sum()
        rc = ones.rolling(f"{window_days}D", closed="left").sum()
        mc_prov_dow_sum[idx2] = rs.values
        mc_prov_dow_cnt[idx2] = rc.values

    rate_cell = mc_cell_sum / np.maximum(1.0, mc_cell_cnt)
    rate_base = mc_prov_dow_sum / np.maximum(1.0, mc_prov_dow_cnt)

    p  = np.clip(rate_base, 1e-4, 1-1e-4)
    n  = np.maximum(1.0, mc_cell_cnt)
    se = np.sqrt(p*(1-p)/n)
    z  = (rate_cell - p) / np.maximum(se, 1e-6)

    d["feat_microclimate_z"] = np.clip(z, -clip_val, clip_val).astype(np.float32)
    return d.drop(columns=["_is_opened"], errors="ignore")

def add_topic_novelty(df, window_days=7, kappa_map=None, tau_map=None):
    d = df.sort_values(["client_id","sent_at"]).copy()
    topic_cols = [c for c in d.columns if c.startswith("camp_topic")]
    if len(topic_cols) == 0:
        d["_topic_key"] = "unknown"
    else:
        mat = d[topic_cols].to_numpy(dtype=int)
        has = (mat.max(axis=1) > 0)
        idx = mat.argmax(axis=1)
        names = np.array([c.replace("camp_topic","") for c in topic_cols], dtype=object)
        d["_topic_key"] = np.where(has, names[idx], "unknown")

    d["tn_sent_local"] = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)

    if kappa_map is None:
        kappa_map = {"email":4.0, "mobile_push":2.0, "sms":1.5, "multichannel":3.0, "unknown":3.0}
    if tau_map is None:
        tau_map = {"email":48.0, "mobile_push":24.0, "sms":72.0, "multichannel":36.0, "unknown":48.0}

    ch = d.get("_channel_raw", "unknown")
    if isinstance(ch, str): ch = pd.Series([ch]*len(d), index=d.index)
    ch = ch.astype(str)

    kappa = ch.map(lambda x: kappa_map.get(x, kappa_map["unknown"])).astype(float).values
    tau_h = ch.map(lambda x: tau_map.get(x, tau_map["unknown"])).astype(float).values

    last_ts_dtype = DatetimeTZDtype(tz=TZ_LOCAL, unit="ns")
    N_recent = pd.Series(0.0, index=d.index, dtype="float32")
    last_ts  = pd.Series(pd.NaT, index=d.index, dtype=last_ts_dtype)

    for (_, _), g in d.groupby(["client_id","_topic_key"], sort=False):
        g2 = g.loc[g["tn_sent_local"].notna()].sort_values("tn_sent_local")
        if g2.empty: continue
        ts = g2["tn_sent_local"]
        ones = pd.Series(1.0, index=ts)
        rc = ones.rolling(f"{window_days}D", closed="left").sum().astype("float32")
        rc.index = g2.index
        N_recent.loc[g2.index] = rc
        last_ts.loc[g2.index]  = ts.shift(1)

    t0 = d["tn_sent_local"].dt.tz_convert("UTC")
    lt = last_ts.dt.tz_convert("UTC")
    t_since_hours = (t0 - lt).dt.total_seconds() / 3600.0
    med = np.nanmedian(t_since_hours)
    if not np.isfinite(med): med = 1e3
    t_since_hours = pd.Series(t_since_hours).fillna(med).clip(lower=0).astype("float32")

    novelty = np.exp(-N_recent.values / np.maximum(kappa, 1e-6)) * \
              (1.0 - np.exp(-t_since_hours.values / np.maximum(tau_h, 1e-6)))

    d["feat_topic_novelty"]  = novelty.astype(np.float32)
    d["topic_N7"]            = N_recent.astype(np.float32)
    d["topic_t_since_hours"] = t_since_hours.astype(np.float32)
    return d

def _bigrams(seq):
    return set(zip(seq[:-1], seq[1:])) if len(seq) >= 2 else set()

def add_path_alignment(df, L=5):
    d = df.sort_values(["client_id","sent_at"]).copy()
    d["_channel_key"] = d["_channel_raw"].fillna("unknown").astype(str)
    out = np.zeros(len(d), dtype=np.float32)

    for uid, g in d.groupby("client_id", sort=False):
        idx = list(g.index)
        chs = g["_channel_key"].tolist()
        lbl = g["is_purchased"].astype(int).tolist()

        history = []
        proto_seq = []
        proto_counts = {}
        align_local = np.zeros(len(idx), dtype=np.float32)

        for j in range(len(idx)):
            recent_seq = history[-L:]
            bg_recent = _bigrams(recent_seq)
            bg_proto  = _bigrams(proto_seq)
            if len(bg_recent)==0 or len(bg_proto)==0:
                align_val = 0.0
            else:
                inter = len(bg_recent & bg_proto)
                uni   = len(bg_recent | bg_proto)
                align_val = float(inter) / float(uni) if uni>0 else 0.0
            align_local[j] = align_val

            history.append(chs[j])
            if lbl[j] == 1:
                seq_before = history[:-1][-L:]
                key = tuple(seq_before)
                proto_counts[key] = proto_counts.get(key, 0) + 1
                if proto_counts:
                    proto_seq = list(max(proto_counts.items(), key=lambda x: x[1])[0])

        out[idx] = align_local

    d["feat_path_align"] = out.astype(np.float32)
    return d

def add_ab_sensitivity_robust_v2(
    df, toggles, ab_col="ab_test",
    ctx_cols=("_channel_raw",),
    a=1.0, b=1.0,
    min_ctx=10, min_var=5,
    tau_user=25.0, tau_ctx=100.0
):
    x = df[["client_id","sent_at","is_purchased","campaign_id"] + list(ctx_cols)].copy()
    x = x.sort_values(["sent_at","client_id"]).reset_index(drop=True)

    ctx_key = x[ctx_cols[0]].astype(str).fillna("unknown")
    for c in ctx_cols[1:]:
        ctx_key = ctx_key + "|" + x[c].astype(str).fillna("unknown")
    x["_ctx"] = ctx_key

    src_ab = np.zeros(len(x), dtype=np.int8)
    src_tg = np.zeros(len(x), dtype=np.int8)
    src_ca = np.zeros(len(x), dtype=np.int8)

    if ab_col in df.columns:
        v0 = df[ab_col].astype(str).str.strip()
        v0 = v0.mask(v0.isin(["","nan","none","null"]))
    else:
        v0 = pd.Series([None]*len(df), index=df.index)

    if toggles and len(toggles) > 0:
        T = df[toggles].fillna(0).astype(int).to_numpy()
        def _toggles_key(row):
            on_idx = np.where(row==1)[0]
            return "tg:" + ",".join(map(str, on_idx)) if len(on_idx) else "tg:none"
        v1 = pd.Series([_toggles_key(r) for r in T], index=df.index)
    else:
        v1 = pd.Series([None]*len(df), index=df.index)

    v2 = pd.Series("camp"+df["campaign_id"].astype(str), index=df.index)

    variant = v0.copy()
    src_ab[variant.notna().values] = 1
    need = variant.isna()
    variant[need] = v1[need]; src_tg[need & v1.notna()] = 1
    need = variant.isna()
    variant[need] = v2[need]; src_ca[need] = 1

    x["_variant"] = variant.values
    x["src_abtest"]   = src_ab
    x["src_toggle"]   = src_tg
    x["src_campaign"] = src_ca

    y = df["is_purchased"].astype(int)

    x["_n_uc"]  = x.groupby(["client_id","_ctx"]).cumcount().astype("float32")
    x["_x_uc"]  = y.groupby([x["client_id"], x["_ctx"]]).cumsum().shift(1).fillna(0.0).astype("float32")
    x["_n_ucv"] = x.groupby(["client_id","_ctx","_variant"]).cumcount().astype("float32")
    x["_x_ucv"] = y.groupby([x["client_id"], x["_ctx"], x["_variant"]]).cumsum().shift(1).fillna(0.0).astype("float32")
    x["_N_c"]   = x.groupby(["_ctx"]).cumcount().astype("float32")
    x["_X_c"]   = y.groupby([x["_ctx"]]).cumsum().shift(1).fillna(0.0).astype("float32")
    x["_N_cv"]  = x.groupby(["_ctx","_variant"]).cumcount().astype("float32")
    x["_X_cv"]  = y.groupby([x["_ctx"], x["_variant"]]).cumsum().shift(1).fillna(0.0).astype("float32")

    x["_n_uc_not"] = (x["_n_uc"] - x["_n_ucv"]).clip(lower=0.0)
    x["_x_uc_not"] = (x["_x_uc"] - x["_x_ucv"]).clip(lower=0.0)
    x["_N_c_not"]  = (x["_N_c"]  - x["_N_cv"]).clip(lower=0.0)
    x["_X_c_not"]  = (x["_X_c"]  - x["_X_cv"]).clip(lower=0.0)

    th_uc_v   = (x["_x_ucv"]   + a) / (x["_n_ucv"]   + a + b)
    th_uc_not = (x["_x_uc_not"]+ a) / (x["_n_uc_not"]+ a + b)
    th_c_v    = (x["_X_cv"]    + a) / (x["_N_cv"]    + a + b)
    th_c_not  = (x["_X_c_not"] + a) / (x["_N_c_not"] + a + b)

    w_user = (x["_n_uc"] / (x["_n_uc"] + tau_user)).fillna(0.0).astype("float32")
    th_v   = w_user * th_uc_v   + (1.0 - w_user) * th_c_v
    th_not = w_user * th_uc_not + (1.0 - w_user) * th_c_not

    var_uc_v   = th_uc_v  *(1-th_uc_v)   / (x["_n_ucv"]   + a + b + 1.0)
    var_uc_not = th_uc_not*(1-th_uc_not) / (x["_n_uc_not"]+ a + b + 1.0)
    var_c_v    = th_c_v   *(1-th_c_v)    / (x["_N_cv"]    + a + b + 1.0)
    var_c_not  = th_c_not *(1-th_c_not)  / (x["_N_c_not"] + a + b + 1.0)
    var_v   = (w_user**2)*var_uc_v   + ((1-w_user)**2)*var_c_v
    var_not = (w_user**2)*var_uc_not + ((1-w_user)**2)*var_c_not

    sens = (th_v - th_not).astype("float32")
    unc  = np.sqrt(var_v + var_not).astype("float32")

    ok_basic = (x["_n_uc"] >= min_ctx)
    ok_user  = (x["_n_ucv"] >= min_var) & (x["_n_uc_not"] >= min_var)
    ok_ctx   = (x["_N_cv"]  >= min_var) & (x["_N_c_not"]  >= min_var)
    gate = (ok_basic & (ok_user | ok_ctx)).astype("float32")

    out = df.copy()
    out["feat_ab_sens"] = (gate * sens).clip(-1.0, 1.0).astype("float32")
    out["feat_ab_unc"]  = (unc / np.maximum(gate, 1e-3)).clip(0.0, 5.0).astype("float32")
    out["feat_ab_mask"] = gate.astype("float32")
    out["src_abtest"]   = x["src_abtest"].values
    out["src_toggle"]   = x["src_toggle"].values
    out["src_campaign"] = x["src_campaign"].values
    return out

def add_user_deliverability_ewma(df, half_life_days=30.0):
    d = df.sort_values(["client_id","sent_at"]).copy()
    HLh = half_life_days*24.0

    def per_user(g):
        last_t = None
        bnc_s, cmp_s, any_s = 0.0, 0.0, 0.0
        out_b, out_c, out_a = [], [], []
        for t, hb, sb, bl, cp in zip(
            g["sent_at"],
            g.get("is_hard_bounced",0), g.get("is_soft_bounced",0),
            g.get("is_blocked",0), g.get("is_complained",0)
        ):
            if pd.isna(t):
                out_b.append(bnc_s); out_c.append(cmp_s); out_a.append(any_s)
                continue
            dt = 0.0 if last_t is None else (t - last_t).total_seconds()/3600.0
            decay = np.exp(-dt/HLh) if dt>=0 else 0.0
            bnc_s *= decay; cmp_s *= decay; any_s *= decay

            out_b.append(bnc_s); out_c.append(cmp_s); out_a.append(any_s)

            bnc_s += 1 if (hb or sb or bl) else 0
            cmp_s += 1 if cp else 0
            any_s += 1 if ((hb or sb or bl) or cp) else 0
            last_t = t

        g["feat_user_deliv_bnc"] = np.array(out_b, dtype=np.float32)
        g["feat_user_deliv_cmp"] = np.array(out_c, dtype=np.float32)
        g["feat_user_deliv_any"] = np.array(out_a, dtype=np.float32)
        return g

    return d.groupby("client_id", group_keys=False).apply(per_user)

def add_toggle_pref_strengths(df, toggles, a=1.0, b=1.0):
    d = df.sort_values(["client_id","sent_at"]).copy()
    if not toggles:
        return d
    y = d["is_purchased"].astype(int)

    for col in toggles:
        if col not in d.columns:
            d[col] = 0
        on  = d[col].fillna(0).astype(int)
        off = 1 - on

        x_on   = (y*on).groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)
        n_on   = (on).groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)
        x_off  = (y*off).groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)
        n_off  = (off).groupby(d["client_id"]).cumsum().shift(1).fillna(0.0)

        th_on  = (x_on  + a)/(n_on  + a + b)
        th_off = (x_off + a)/(n_off + a + b)

        sens = (th_on - th_off).astype("float32")
        var_on  = th_on *(1-th_on )/(n_on  + a + b + 1.0)
        var_off = th_off*(1-th_off)/(n_off + a + b + 1.0)
        unc = np.sqrt(var_on + var_off).astype("float32")

        key = col.replace("subject_with_","tg_")
        d[f"feat_{key}_sens"] = sens.clip(-1,1)
        d[f"feat_{key}_unc"]  = unc.clip(0,1)

    return d

def add_like_last_success(df, use_toggles=True):
    d = df.sort_values(["client_id","sent_at"]).copy()
    attr_cols = [c for c in d.columns if c.startswith(("camp_campaign_type","camp_channel","camp_topic"))]
    if use_toggles:
        attr_cols += [c for c in d.columns if c.startswith("subject_with_")]
    attr_cols = sorted(set(attr_cols))

    vec = d[attr_cols].fillna(0).astype(int).to_numpy()
    out = np.zeros(len(d), dtype=np.float32)
    last_vec = {}

    for i, (uid, y) in enumerate(zip(d["client_id"].values, d["is_purchased"].astype(int).values)):
        v = vec[i]
        if uid in last_vec:
            a = last_vec[uid]
            inter = (np.minimum(a, v).sum())
            uni   = (np.maximum(a, v).sum())
            out[i] = float(inter)/float(uni) if uni>0 else 0.0
        else:
            out[i] = 0.0
        if y == 1:
            last_vec[uid] = v.copy()

    d["feat_like_last_success"] = out.astype(np.float32)
    return d

def add_channel_recency_counts(df, channels=("email","mobile_push","sms","multichannel"), compute_counts=False):
    d = df.sort_values(["client_id","sent_at"]).copy()
    d["sent_at_local"] = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)

    for ch in channels:
        key = d["_channel_raw"].fillna("unknown").astype(str)
        mask = (key == ch)
        last_t = d["sent_at"].where(mask).groupby(d["client_id"]).ffill().shift(1)

        fallback = np.nanmedian((d["sent_at"]-d["sent_at"].min()).dt.total_seconds()/3600.0)
        if not np.isfinite(fallback): fallback = 24.0

        d[f"feat_last_{ch}_hours"] = ((d["sent_at"] - last_t).dt.total_seconds()/3600.0) \
                                        .fillna(fallback).clip(lower=0).astype(np.float32)

        if compute_counts:
            for H in (24,72):
                cnt = np.zeros(len(d), dtype=np.float32)
                for uid, g in d.loc[mask].groupby("client_id", sort=False):
                    g2 = g.loc[g["sent_at_local"].notna()].sort_values("sent_at_local")
                    if g2.empty: continue
                    ts = g2["sent_at_local"]; ones = pd.Series(1.0, index=ts)
                    r = ones.rolling(f"{H}H", closed="left").sum()
                    cnt[g2.index] = r.values
                d[f"feat_cnt_{ch}_{H}h"] = cnt

    return d.drop(columns=["sent_at_local"], errors="ignore")

def add_payday_monthend_proximity(df, payday=25, sigma_pay=2.0, sigma_me=1.5):
    d = df.copy()
    local = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)

    dom = local.dt.day.astype("float32")
    eom = (local + pd.offsets.MonthEnd(0)).dt.day.astype("float32")
    days_to_eom = (eom - dom)
    diff_pay    = (dom - float(payday)).abs()

    med_eom = np.nanmedian(days_to_eom)
    med_pay = np.nanmedian(diff_pay)
    if not np.isfinite(med_eom): med_eom = 15.0
    if not np.isfinite(med_pay): med_pay = 10.0

    days_to_eom = pd.Series(days_to_eom).fillna(med_eom).clip(lower=0).astype("float32")
    diff_pay    = pd.Series(diff_pay).fillna(med_pay).astype("float32")

    d["feat_payday_bump"]   = np.exp(-(diff_pay**2)    /(2.0*sigma_pay**2)).astype("float32")
    d["feat_monthend_bump"] = np.exp(-(days_to_eom**2)/(2.0*sigma_me**2)).astype("float32")
    return d

def add_provider_health_rolling(df, win_short=7, win_long=30, clip_eps=1e-4):
    d = df.sort_values("sent_at").copy()
    prov = d.get("email_provider_raw", "other")
    if isinstance(prov, str): prov = pd.Series([prov]*len(d), index=d.index)
    d["_prov_key"] = prov.fillna("other").astype(str)

    ts_local = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)
    d["_ts_local"] = ts_local

    opened     = pd.to_numeric(d.get("is_opened", 0), errors="coerce").fillna(0).astype(int)
    complained = pd.to_numeric(d.get("is_complained", 0), errors="coerce").fillna(0).astype(int)

    hb = pd.to_numeric(d.get("is_hard_bounced", 0), errors="coerce").fillna(0).astype(int)
    sb = pd.to_numeric(d.get("is_soft_bounced", 0), errors="coerce").fillna(0).astype(int)
    bl = pd.to_numeric(d.get("is_blocked", 0), errors="coerce").fillna(0).astype(int)
    bounced = ((hb>0)|(sb>0)|(bl>0)).astype(int)

    r_open_s = np.zeros(len(d), np.float32); r_open_l = np.zeros(len(d), np.float32)
    r_bnc_s  = np.zeros(len(d), np.float32); r_bnc_l  = np.zeros(len(d), np.float32)
    r_cmp_s  = np.zeros(len(d), np.float32); r_cmp_l  = np.zeros(len(d), np.float32)

    for _, g in d.groupby("_prov_key", sort=False):
        g2 = g.loc[g["_ts_local"].notna()].sort_values("_ts_local")
        if g2.empty: continue
        idx2 = g2.index; ts = g2["_ts_local"]

        ones = pd.Series(1.0, index=ts)
        o = pd.Series(opened.loc[idx2].astype(float).values, index=ts)
        b = pd.Series(bounced.loc[idx2].astype(float).values, index=ts)
        c = pd.Series(complained.loc[idx2].astype(float).values, index=ts)

        n_s = ones.rolling(f"{win_short}D", closed="left").sum()
        n_l = ones.rolling(f"{win_long}D",  closed="left").sum()
        o_s = o.rolling(f"{win_short}D", closed="left").sum()
        o_l = o.rolling(f"{win_long}D",  closed="left").sum()
        b_s = b.rolling(f"{win_short}D", closed="left").sum()
        b_l = b.rolling(f"{win_long}D",  closed="left").sum()
        c_s = c.rolling(f"{win_short}D", closed="left").sum()
        c_l = c.rolling(f"{win_long}D",  closed="left").sum()

        r_open_s[idx2] = (o_s / np.maximum(n_s, 1.0)).fillna(0.0).values
        r_open_l[idx2] = (o_l / np.maximum(n_l, 1.0)).fillna(0.0).values
        r_bnc_s[idx2]  = (b_s / np.maximum(n_s, 1.0)).fillna(0.0).values
        r_bnc_l[idx2]  = (b_l / np.maximum(n_l, 1.0)).fillna(0.0).values
        r_cmp_s[idx2]  = (c_s / np.maximum(n_s, 1.0)).fillna(0.0).values
        r_cmp_l[idx2]  = (c_l / np.maximum(n_l, 1.0)).fillna(0.0).values

    clip01 = lambda a: np.clip(a, clip_eps, 1.0-clip_eps).astype(np.float32)
    d["prov_open7"]        = clip01(r_open_s)
    d["prov_open30"]       = clip01(r_open_l)
    d["prov_open_delta"]   = (d["prov_open7"]  - d["prov_open30"]).clip(-1,1).astype(np.float32)
    d["prov_bounce7"]      = clip01(r_bnc_s)
    d["prov_bounce30"]     = clip01(r_bnc_l)
    d["prov_bounce_delta"] = (d["prov_bounce7"]- d["prov_bounce30"]).clip(-1,1).astype(np.float32)
    d["prov_comp7"]        = clip01(r_cmp_s)
    d["prov_comp30"]       = clip01(r_cmp_l)
    d["prov_comp_delta"]   = (d["prov_comp7"] - d["prov_comp30"]).clip(-1,1).astype(np.float32)
    return d.drop(columns=["_prov_key","_ts_local"], errors="ignore")

def add_user_recent_behavior(df, wins=(7,30)):
    d = df.sort_values(["client_id","sent_at"]).copy()
    d["sent_at_local2"] = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)

    opened = pd.to_numeric(d.get("is_opened",0), errors="coerce").fillna(0).astype(int)
    clicked= pd.to_numeric(d.get("is_clicked",0), errors="coerce").fillna(0).astype(int)
    bought  = pd.to_numeric(d.get("is_purchased",0), errors="coerce").fillna(0).astype(int)

    for uid, g in d.groupby("client_id", sort=False):
        g = g.loc[g["sent_at_local2"].notna()].sort_values("sent_at_local2")
        if g.empty: continue
        idx = g.index; ts = g["sent_at_local2"]

        ones = pd.Series(1.0, index=ts)
        o = pd.Series(opened.loc[idx].astype(float).values,  index=ts)
        c = pd.Series(clicked.loc[idx].astype(float).values, index=ts)
        p = pd.Series(bought.loc[idx].astype(float).values,  index=ts)

        for W in wins:
            nW = ones.rolling(f"{W}D", closed="left").sum()
            d.loc[idx, f"u_open_cnt_{W}d"]  = o.rolling(f"{W}D", closed="left").sum().values
            d.loc[idx, f"u_click_cnt_{W}d"] = c.rolling(f"{W}D", closed="left").sum().values
            d.loc[idx, f"u_buy_cnt_{W}d"]   = p.rolling(f"{W}D", closed="left").sum().values

            denom = np.maximum(nW.values, 1.0)
            d.loc[idx, f"u_open_rate_{W}d"]  = (d.loc[idx, f"u_open_cnt_{W}d"] / denom)
            d.loc[idx, f"u_click_rate_{W}d"] = (d.loc[idx, f"u_click_cnt_{W}d"] / denom)
            d.loc[idx, f"u_buy_rate_{W}d"]   = (d.loc[idx, f"u_buy_cnt_{W}d"] / denom)

    for col in [c for c in d.columns if c.startswith(("u_open_","u_click_","u_buy_"))]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype("float32")

    return d.drop(columns=["sent_at_local2"], errors="ignore")

def add_context_perf_rolling(df, wins=(7,30), clip=1e-4):
    d = df.sort_values("sent_at").copy()

    if "_topic_key" not in d.columns:
        topic_cols = [c for c in d.columns if c.startswith("camp_topic")]
        if topic_cols:
            mat = d[topic_cols].to_numpy(dtype=int)
            names = np.array([c.replace("camp_topic","") for c in topic_cols], dtype=object)
            has = (mat.max(axis=1) > 0); idx = mat.argmax(axis=1)
            d["_topic_key"] = np.where(has, names[idx], "unknown")
        else:
            d["_topic_key"] = "unknown"

    d["_ctx_tc"] = d["_topic_key"].astype(str) + "|" + d.get("_channel_raw","unknown").astype(str)
    d["sent_at_local_cp"] = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)

    opened = pd.to_numeric(d.get("is_opened",0), errors="coerce").fillna(0).astype(int)
    bought = pd.to_numeric(d.get("is_purchased",0), errors="coerce").fillna(0).astype(int)

    def _roll(keycol, prefix):
        for _, g in d.groupby(keycol, sort=False):
            g2 = g.loc[g["sent_at_local_cp"].notna()].sort_values("sent_at_local_cp")
            if g2.empty: continue
            idx = g2.index; ts = g2["sent_at_local_cp"]

            ones = pd.Series(1.0, index=ts)
            o = pd.Series(opened.loc[idx].astype(float).values, index=ts)
            p = pd.Series(bought.loc[idx].astype(float).values, index=ts)

            for W in wins:
                nW = ones.rolling(f"{W}D", closed="left").sum()
                orate = (o.rolling(f"{W}D", closed="left").sum()/np.maximum(nW,1.0)).clip(clip,1-clip)
                prate = (p.rolling(f"{W}D", closed="left").sum()/np.maximum(nW,1.0)).clip(clip,1-clip)
                d.loc[idx, f"{prefix}_open_rate_{W}d"] = orate.values
                d.loc[idx, f"{prefix}_buy_rate_{W}d"]  = prate.values

    _roll("_ctx_tc","ctx_tc")
    _roll("campaign_id","ctx_camp")

    for c in d.columns:
        if c.startswith(("ctx_tc_","ctx_camp_")):
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("float32")

    return d.drop(columns=["sent_at_local_cp"], errors="ignore")

def add_last_any_hours(df):
    d = df.sort_values(["client_id","sent_at"]).copy()
    last_any = d.groupby("client_id")["sent_at"].shift(1)
    fallback = np.nanmedian((d["sent_at"]-d["sent_at"].min()).dt.total_seconds()/3600.0)
    if not np.isfinite(fallback): fallback = 24.0
    d["feat_last_any_hours"] = ((d["sent_at"] - last_any).dt.total_seconds()/3600.0) \
                                .fillna(fallback).clip(lower=0).astype("float32")
    return d

def add_cadence_std_30d(df):
    d = df.sort_values(["client_id","sent_at"]).copy()
    d["sent_at_local3"] = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)
    out = np.zeros(len(d), dtype=np.float32)

    for uid, g in d.groupby("client_id", sort=False):
        g2 = g.loc[g["sent_at_local3"].notna()].sort_values("sent_at_local3")
        if len(g2) < 3:
            continue

        ts = g2["sent_at_local3"]
        dt_h_vals = ts.diff().dt.total_seconds() / 3600.0
        dt_h = pd.Series(dt_h_vals.values, index=ts)

        std_30d = dt_h.rolling("30D", closed="left").std().fillna(0.0).astype(np.float32)
        out[g2.index] = std_30d.reindex(ts).fillna(0.0).values

    d["u_cadence_std_30d"] = out
    return d.drop(columns=["sent_at_local3"], errors="ignore")

def add_calendar_extras(df, sigma_eoq=1.5):
    d = df.copy()
    local = pd.to_datetime(d["sent_at"], utc=True).dt.tz_convert(TZ_LOCAL)
    d["cal_is_weekend"] = (local.dt.dayofweek>=5).astype("int32")
    wom = ((local.dt.day-1)//7 + 1).clip(1,5).astype("Int16")
    d["cal_week_of_month"] = wom
    eoq = (local + pd.offsets.QuarterEnd(0)).dt.day
    diff_eoq = (eoq - local.dt.day).astype("float32").clip(lower=0)
    d["feat_eoq_bump"] = np.exp(-(diff_eoq**2)/(2.0*sigma_eoq**2)).astype("float32")
    return d

def add_purchase_refractory(df, tau_hours=72.0):
    d = df.copy()
    last_buy = d["purchased_at"].where(d["is_purchased"]==1).groupby(d["client_id"]).ffill().shift(1)
    dt_h = (pd.to_datetime(d["sent_at"],utc=True)-pd.to_datetime(last_buy,utc=True)).dt.total_seconds()/3600.0
    dt_h = pd.to_numeric(dt_h, errors="coerce")
    med = float(np.nanmedian(dt_h)) if np.isfinite(np.nanmedian(dt_h)) else 1e3
    dt_h = dt_h.fillna(med).clip(lower=0)
    d["feat_postbuy_refrac"] = np.exp(-dt_h/max(tau_hours,1e-6)).astype("float32")
    return d

# =====================================================
# 4) RUN PIPELINE
# =====================================================
def build_final_data(mode=MODE):
    final_data, ecc_toggle_cols, EP_TOP3 = load_and_preprocess(mode=mode)

    # --- base/v1~v4 + extras ---
    final_data = add_ready_to_buy_hazard(final_data)
    final_data = add_hour_shift_delta(final_data)
    final_data = add_dow_shift_delta(final_data)
    final_data = add_fatigue_and_cooldown(final_data)
    final_data = add_ecc_distance(final_data, ecc_toggle_cols)
    final_data = add_microclimate_z_v2(final_data)
    final_data = add_topic_novelty(final_data)
    final_data = add_path_alignment(final_data, L=5)
    final_data = add_ab_sensitivity_robust_v2(final_data, ecc_toggle_cols, "ab_test", ("_channel_raw",))
    final_data = add_toggle_pref_strengths(final_data, ecc_toggle_cols)
    final_data = add_like_last_success(final_data, use_toggles=True)
    final_data = add_user_deliverability_ewma(final_data)
    final_data = add_channel_recency_counts(final_data, compute_counts=False)
    final_data = add_payday_monthend_proximity(final_data)
    final_data = add_provider_health_rolling(final_data)

    # === New Features ===
    final_data = add_user_recent_behavior(final_data)
    final_data = add_context_perf_rolling(final_data)
    final_data = add_last_any_hours(final_data)
    final_data = add_cadence_std_30d(final_data)
    final_data = add_calendar_extras(final_data)
    final_data = add_purchase_refractory(final_data)

    return final_data, ecc_toggle_cols, EP_TOP3

if __name__ == "__main__":
    print("Running preprocessing pipeline...")
    final_data, ecc_toggle_cols, EP_TOP3 = build_final_data(mode=MODE)
    print("final_data shape:", final_data.shape)
    
    output_path = os.path.join(DATA_DIR, "featured_all.parquet")
    print(f"Saving to {output_path}...")
    final_data.to_parquet(output_path)
    print("Saved!")
