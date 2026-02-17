import pandas as pd
import numpy as np
from scipy import stats
import json
import itertools

# --- 1. CONFIGURAZIONE FLESSIBILE ---
# Se metti una stringa, usa il nome del CSV.
# Se metti una tupla ("Nome_CSV", "Nome_Dashboard"), rimappa il nome.
VARIABLE_GROUPS = {
    "Generali": [
        ("raven risposte corrette", "Raven"),
        ("denominazione", "Denominazione (BVL)"),
        ("articolazione", "Articolazione (BVL)"),
        ("ripetizione frasi", "Ripetizione Frasi (BVL)"),
        ("ripetizione NP", "Ripetizione Non Parole (BVL)"),
        ("totale fluenza", "Fluenza semantica (TNL)"),
    ],
    "Produttività": [
        ("tempo descrittivo MMS_FA", "Tempo Descrittivo"),
        ("MLU_units", "LME (Unità)"),
        ("MLU_words", "LME (Parole)"),
        ("totale parole", "Parole Totali"),
        "% mono-bisillabiche", 
        "% trisillabiche", 
        "% polisillabiche"
    ],
    "Accuratezza": [
        ("% parole corrette", "Parole Corrette (%)"),
        ("% parole solo processi", "Processi Fonologici (%)"),
        ("‰ idios", "Idiosincrasie (‰)"),
        ("% variabilità", "Variabilità (%)"),
        ("%globalAccuracy", "Accuratezza Globale (%)"),
        ("%consonantAccuracy", "Accuratezza Consonantica (%)"),
        ("% distortions", "Distorsioni (%)"),
        ("% falsestart", "False Partenze (%)"),
    ],
    "Modo Articolazione": [
        "% occlusive", "% fricative", "% affricate", 
        "% nasali", "% laterali", "% polivibranti"
    ],
    "Luogo Articolazione": [
        "% bilabiali", "% labio-dentali", "% dentali", 
        "% alveolari", "% postalveolari", "% palatali", "% velari"
    ],
    "Inventario": [
        "fonemi presenti", "fonemi emergenti", "fonemi assenti",
        ("% fonemi presenti", "Completezza Inventario (%)")
    ],
    "Metriche sperimentali": [
        ("TTR * 100", "TTR (%)"),
        ("HDD * 100", "HDD (%)"),
        ("MATTR50 * 100", "MATTR50 (%)"),
        ("MATTR5% * 100", "MATTR5% (%)"),
        ("MTLD", "MTLD"),
        ("WIM", "WIM"),
        ("vocD", "vocD"),
        ("lexicalDensity (content/words) * 100", "Lexical Density (%)"),
        ("overall similarity * 100", "Overall Similarity (%)")
    ],
    
}


ORDER_MAP = {
    'annuale': ["3.6-4.5", "4.6-5.5", "5.6-6.5"],
    'semestrale': ["3.6-3.11", "4.0-4.5", "4.6-4.11", "5.0-5.5", "5.6-5.11", "6.0-6.5"]
}

def clean_and_load():
    try:
        df = pd.read_csv("normative_data.csv", dtype=str)
        df.columns = df.columns.str.strip()
        csv_cols = []
        for items in VARIABLE_GROUPS.values():
            for item in items:
                csv_cols.append(item[0] if isinstance(item, tuple) else item)
        for col in set(csv_cols + ['eta mesi']):
            if col in df.columns:
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Errore caricamento: {e}")
        return pd.DataFrame()

def calculate_stats(df, metric, group_col, mode_key):
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    data = df[[group_col, metric]].dropna()
    custom_order = ORDER_MAP.get(mode_key, [])
    present_groups = [g for g in custom_order if g in data[group_col].unique()]
    name_to_num = {name: str(i+1) for i, name in enumerate(present_groups)}
    if not present_groups: return None
    groups_list = [data[data[group_col] == g][metric] for g in present_groups]
    anova_res = {"sig": False, "text": "N/A"}
    if len(groups_list) >= 2:
        try:
            f, p = stats.f_oneway(*groups_list)
            anova_res = {"sig": bool(p < 0.05), "text": f"F={f:.2f}, p={p:.4f}"}
        except: pass
    posthoc_res = []
    if anova_res["sig"]:
        pairs = list(itertools.combinations(present_groups, 2))
        corr = len(pairs)
        for g1, g2 in pairs:
            d1, d2 = data[data[group_col] == g1][metric], data[data[group_col] == g2][metric]
            try:
                t, p_ph = stats.ttest_ind(d1, d2, equal_var=False)
                p_adj = min(p_ph * corr, 1.0)
                if p_adj < 0.05:
                    posthoc_res.append(f"<b>{name_to_num[g1]} vs {name_to_num[g2]}</b> (p={p_adj:.3f})")
            except: continue
    stats_table = []
    for g in present_groups:
        vals = data[data[group_col] == g][metric]
        # CALCOLO PERCENTILI
        p = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
        
        stats_table.append({
            "Fascia": g, 
            "ID": name_to_num[g], 
            "N": int(len(vals)),
            "Media": round(float(vals.mean()), 2), 
            "DS": round(float(vals.std()), 2) if len(vals) > 1 else 0,
            "P5": round(float(p[0]), 2),
            "P10": round(float(p[1]), 2),
            "P25": round(float(p[2]), 2),
            "P50": round(float(p[3]), 2),
            "P75": round(float(p[4]), 2),
            "P90": round(float(p[5]), 2),
            "P95": round(float(p[6]), 2)
        })

    return {
        "name": metric, "anova": anova_res, "posthoc": posthoc_res, "table": stats_table,
        "chart_data": [{"group": name_to_num[g], "values": data[data[group_col] == g][metric].tolist()} for g in present_groups]
    }

# --- ESECUZIONE ---
df = clean_and_load()
output_data = {"annuale": {}, "semestrale": {}, "descrittive": {}, "info_campione": {}}
output_data["info_campione"] = {
    "totale": int(len(df)),
    "eta": {"media": round(float(df['eta mesi'].mean()), 1), "min": int(df['eta mesi'].min()), "max": int(df['eta mesi'].max())}
}
for mode in ["annuale", "semestrale"]:
    col = 'fascia eta 12m' if mode == 'annuale' else 'fascia eta 6m'
    output_data["descrittive"][mode] = [] 
    custom_order = ORDER_MAP.get(mode, [])
    for i, g in enumerate(custom_order):
        if g in df[col].unique():
            sub = df[df[col] == g]
            output_data["descrittive"][mode].append({
                "fascia": i+1, 
                "M": int((sub['sesso'].str.upper() == 'M').sum()), 
                "F": int((sub['sesso'].str.upper() == 'F').sum())
            })
    for cat, items in VARIABLE_GROUPS.items():
        res_list = []
        for item in items:
            csv_n, disp_n = (item[0], item[1]) if isinstance(item, tuple) else (item, item)
            if csv_n in df.columns:
                res = calculate_stats(df, csv_n, col, mode)
                if res:
                    res['name'] = disp_n.upper()
                    res_list.append(res)
        if res_list:
            output_data[mode][cat] = res_list

with open("data.js", "w", encoding="utf-8") as f:
    f.write("const GLOBAL_DATA = " + json.dumps(output_data, ensure_ascii=False, indent=2) + ";")

print("\n✅ Generazione completata.")