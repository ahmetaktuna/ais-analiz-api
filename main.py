from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

app = FastAPI()

# Web sitenizden gelecek isteklere (CORS) izin veriyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Güvenlik için daha sonra buraya sadece sitenizin adını yazabiliriz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    depVar: str
    indepVars: List[str]
    data: List[Dict[str, Any]]

# UYKU ENGELLEYİCİ PING ADRESİMİZ
@app.get("/ping")
def ping():
    return {"status": "Uyanigim ve hazirim!"}

# REGRESYON HESAPLAMA MOTORU
@app.post("/analyze")
def analyze(req: AnalysisRequest):
    try:
        df = pd.DataFrame(req.data)
        
        # Seçili sütunları sayısala çevir ve boş verileri sil
        cols = [req.depVar] + req.indepVars
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=cols)
        
        Y = df[req.depVar]
        X = df[req.indepVars]
        X = sm.add_constant(X) # Sabit terim
        
        n = len(df)
        k = len(req.indepVars)
        
        # Python Statsmodels ile OLS Analizi
        model = sm.OLS(Y, X).fit()
        
        # VIF ve Tolerance Hesaplama
        vifs, tolerances = [None], [None] # Sabit terim için None
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
                    
        # Standartlaştırılmış Beta Hesaplama
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
            
        result = {
            "n": n,
            "k": k,
            "R2": float(model.rsquared),
            "adjR2": float(model.rsquared_adj),
            "F": float(model.fvalue),
            "df_model": float(model.df_model),
            "df_error": float(model.df_resid),
            "p_F": float(model.f_pvalue),
            "DW": float(durbin_watson(model.resid)),
            "coeffData": coeffData,
            "depVar": req.depVar,
            "indepVars": req.indepVars
        }
        
        return result
    except Exception as e:
        return {"error": str(e)}