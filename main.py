from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

app = FastAPI()

# Web sitenizden gelecek isteklere (CORS) izin veriyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    depVar: str
    indepVars: List[str]
    data: List[Dict[str, Any]]

class TTestRequest(BaseModel):
    depVar: str
    groupVar: str
    data: List[Dict[str, Any]]

# UYKU ENGELLEYİCİ PING ADRESİMİZ
@app.get("/ping")
def ping():
    return {"status": "Uyanigim ve hazirim!"}

# ==========================================
# 1. REGRESYON HESAPLAMA MOTORU
# ==========================================
@app.post("/analyze")
def analyze(req: AnalysisRequest):
    try:
        df = pd.DataFrame(req.data)
        cols = [req.depVar] + req.indepVars
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=cols)
        
        Y = df[req.depVar]
        X = df[req.indepVars]
        X = sm.add_constant(X)
        
        n = len(df)
        k = len(req.indepVars)
        
        model = sm.OLS(Y, X).fit()
        
        vifs, tolerances = [None], [None]
        if k == 1:
            vifs.append(1.0)
            tolerances.append(1.0)
        else:
            for i in range(1, X.shape[1]):
                try:
                    v = variance_inflation_factor(X.values, i)
                    vifs.append(float(v) if not np.isinf(v) else 999.0)
                    tolerances.append(float(1/v) if v != 0 else 0.001)
                except:
                    vifs.append(999.0)
                    tolerances.append(0.001)
                    
        sd_y = Y.std(ddof=1)
        betas = [None]
        for indep in req.indepVars:
            sd_x = df[indep].std(ddof=1)
            b_unstd = model.params[indep]
            betas.append(float(b_unstd * (sd_x / sd_y)))
            
        coeffData = []
        vars_list = ['const'] + req.indepVars
        display_names = ['Sabit Terim'] + req.indepVars
        
        for i, (var, d_name) in enumerate(zip(vars_list, display_names)):
            coeffData.append({
                "name": d_name,
                "B": float(model.params[var]),
                "SE": float(model.bse[var]),
                "Beta": betas[i],
                "t": float(model.tvalues[var]),
                "p": float(model.pvalues[var]),
                "Tol": tolerances[i],
                "VIF": vifs[i]
            })
            
        return {
            "n": n, "k": k, "R2": float(model.rsquared), "adjR2": float(model.rsquared_adj),
            "F": float(model.fvalue), "df_model": float(model.df_model), "df_error": float(model.df_resid),
            "p_F": float(model.f_pvalue), "DW": float(durbin_watson(model.resid)),
            "coeffData": coeffData, "depVar": req.depVar, "indepVars": req.indepVars
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 2. BAĞIMSIZ ÖRNEKLEM T-TEST MOTORU
# ==========================================
@app.post("/ttest-independent")
def ttest_independent(req: TTestRequest):
    try:
        df = pd.DataFrame(req.data)
        
        # Bağımlı değişkeni sayısal yap, boşları sil
        df[req.depVar] = pd.to_numeric(df[req.depVar], errors='coerce')
        df = df.dropna(subset=[req.depVar, req.groupVar])
        
        # Grupları belirle
        unique_groups = df[req.groupVar].unique()
        if len(unique_groups) != 2:
            return {"error": f"Bağımsız örneklem t-testi için grup değişkeninizde tam olarak 2 farklı kategori olmalıdır. Sizde {len(unique_groups)} kategori bulundu: {list(unique_groups)}"}
        
        g1_name, g2_name = unique_groups[0], unique_groups[1]
        data1 = df[df[req.groupVar] == g1_name][req.depVar].values
        data2 = df[df[req.groupVar] == g2_name][req.depVar].values
        
        n1, n2 = len(data1), len(data2)
        if n1 < 2 or n2 < 2:
            return {"error": "Gruplardan birinde yeterli veri yok (En az 2 veri olmalı)."}

        # 1. Betimsel İstatistikler
        desc = {
            "g1": {"name": str(g1_name), "n": n1, "mean": float(np.mean(data1)), "std": float(np.std(data1, ddof=1)), "se": float(np.std(data1, ddof=1)/np.sqrt(n1))},
            "g2": {"name": str(g2_name), "n": n2, "mean": float(np.mean(data2)), "std": float(np.std(data2, ddof=1)), "se": float(np.std(data2, ddof=1)/np.sqrt(n2))}
        }
        
        # 2. Varsayım Testleri
        # Normallik (Shapiro-Wilk)
        try:
            stat_s1, p_s1 = stats.shapiro(data1)
            stat_s2, p_s2 = stats.shapiro(data2)
        except:
            stat_s1, p_s1, stat_s2, p_s2 = None, None, None, None
            
        # Homojenlik (Levene Test)
        try:
            stat_lev, p_lev = stats.levene(data1, data2, center='mean')
        except:
            stat_lev, p_lev = None, None

        # 3. T-Testleri (Eşit ve Eşit Olmayan Varyanslar)
        t_eq, p_eq = stats.ttest_ind(data1, data2, equal_var=True)
        df_eq = n1 + n2 - 2
        
        t_uneq, p_uneq = stats.ttest_ind(data1, data2, equal_var=False)
        # Welch Satterthwaite df
        v1 = np.var(data1, ddof=1)
        v2 = np.var(data2, ddof=1)
        num = (v1/n1 + v2/n2)**2
        den = (v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)
        df_uneq = num/den if den != 0 else df_eq
        
        # 4. Cohen's d (Effect Size)
        # pooled std
        pooled_std = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std != 0 else 0

        return {
            "depVar": req.depVar,
            "groupVar": req.groupVar,
            "descriptives": desc,
            "assumptions": {
                "shapiro": {"g1": {"W": stat_s1, "p": p_s1}, "g2": {"W": stat_s2, "p": p_s2}},
                "levene": {"F": stat_lev, "p": p_lev}
            },
            "ttest": {
                "equal_var": {"t": float(t_eq), "df": float(df_eq), "p": float(p_eq)},
                "unequal_var": {"t": float(t_uneq), "df": float(df_uneq), "p": float(p_uneq)}
            },
            "cohens_d": float(abs(cohens_d))
        }
    except Exception as e:
        return {"error": str(e)}
