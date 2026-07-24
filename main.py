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
import itertools
import math

app = FastAPI()

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

class MultipleTTestRequest(BaseModel):
    depVars: List[str]
    groupVar: str
    data: List[Dict[str, Any]]

class MultipleANOVARequest(BaseModel):
    depVars: List[str]
    groupVar: str
    postHoc: str
    data: List[Dict[str, Any]]

class NormalityRequest(BaseModel):
    depVars: List[str]
    data: List[Dict[str, Any]]

@app.get("/ping")
def ping():
    return {"status": "Uyanığım ve analize hazırım!"}

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    try:
        df = pd.DataFrame(req.data)
        cols = [req.depVar] + req.indepVars
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=cols)
        
        if len(df) < 3:
            return {"error": "Geçerli veri sayısı çok az (En az 3 satır gerekli)."}

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
                    vifs.append(float(v) if not np.isinf(v) and not np.isnan(v) else 999.0)
                    tolerances.append(float(1/v) if v != 0 and not np.isinf(v) and not np.isnan(v) else 0.001)
                except:
                    vifs.append(999.0)
                    tolerances.append(0.001)
                    
        sd_y = Y.std(ddof=1)
        if pd.isna(sd_y) or sd_y == 0:
            sd_y = 1.0 # Sıfıra bölünme hatasını engellemek için
            
        betas = [None]
        for indep in req.indepVars:
            sd_x = df[indep].std(ddof=1)
            if pd.isna(sd_x): sd_x = 0
            b_unstd = model.params[indep]
            betas.append(float(b_unstd * (sd_x / sd_y)))
            
        coeffData = []
        vars_list = ['const'] + req.indepVars
        display_names = ['Sabit Terim'] + req.indepVars
        
        for i, (var, d_name) in enumerate(zip(vars_list, display_names)):
            coeffData.append({
                "name": d_name, "B": float(model.params[var]), "SE": float(model.bse[var]),
                "Beta": betas[i], "t": float(model.tvalues[var]), "p": float(model.pvalues[var]),
                "Tol": tolerances[i], "VIF": vifs[i]
            })
            
        return {
            "n": n, "k": k, "R2": float(model.rsquared), "adjR2": float(model.rsquared_adj),
            "F": float(model.fvalue) if not np.isnan(model.fvalue) else 0.0, 
            "df_model": float(model.df_model), "df_error": float(model.df_resid),
            "p_F": float(model.f_pvalue) if not np.isnan(model.f_pvalue) else 1.0, 
            "DW": float(durbin_watson(model.resid)),
            "coeffData": coeffData, "depVar": req.depVar, "indepVars": req.indepVars
        }
    except Exception as e:
        return {"error": f"Regresyon Hatası: {str(e)}"}

@app.post("/ttest-multiple")
def ttest_multiple(req: MultipleTTestRequest):
    try:
        df = pd.DataFrame(req.data)
        df = df.dropna(subset=[req.groupVar])
        
        unique_groups = df[req.groupVar].unique()
        if len(unique_groups) != 2:
            return {"error": f"Grup değişkeninizde tam olarak 2 kategori olmalıdır. Sizde {len(unique_groups)} bulundu."}
        
        g1_val, g2_val = unique_groups[0], unique_groups[1]
        
        results = []
        for var in req.depVars:
            temp_df = df.dropna(subset=[var])
            data1 = temp_df[temp_df[req.groupVar] == g1_val][var]
            data2 = temp_df[temp_df[req.groupVar] == g2_val][var]
            
            data1 = pd.to_numeric(data1, errors='coerce').dropna().values
            data2 = pd.to_numeric(data2, errors='coerce').dropna().values
            
            n1, n2 = len(data1), len(data2)
            if n1 < 2 or n2 < 2:
                continue 
                
            m1, m2 = float(np.mean(data1)), float(np.mean(data2))
            std1 = float(np.std(data1, ddof=1)) if n1 > 1 else 0.0
            std2 = float(np.std(data2, ddof=1)) if n2 > 1 else 0.0
            
            try:
                stat_lev, p_lev = stats.levene(data1, data2, center='mean')
            except:
                p_lev = 1.0 
                
            is_equal_var = p_lev >= 0.05 if not np.isnan(p_lev) else True
            
            try:
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=is_equal_var)
            except:
                t_stat, p_val = 0.0, 1.0
                
            v1, v2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
            cohens_d = abs(m1 - m2) / pooled_std if pooled_std > 0 else 0.0
            
            results.append({
                "varName": var,
                "is_equal_var": bool(is_equal_var),
                "levene_p": float(p_lev) if not np.isnan(p_lev) else 1.0,
                "t": float(t_stat) if not np.isnan(t_stat) else 0.0,
                "p": float(p_val) if not np.isnan(p_val) else 1.0,
                "cohens_d": float(cohens_d),
                "g1": {"val": str(g1_val), "n": n1, "mean": m1, "std": std1},
                "g2": {"val": str(g2_val), "n": n2, "mean": m2, "std": std2}
            })

        return {
            "groupVar": req.groupVar,
            "originalGroups": [str(g1_val), str(g2_val)],
            "results": results
        }
    except Exception as e:
        return {"error": f"T-Testi Hatası: {str(e)}"}

@app.post("/anova-multiple")
def anova_multiple(req: MultipleANOVARequest):
    try:
        df = pd.DataFrame(req.data)
        df = df.dropna(subset=[req.groupVar])
        
        unique_groups = list(df[req.groupVar].unique())
        unique_groups.sort(key=lambda x: str(x))
        
        if len(unique_groups) < 2:
            return {"error": f"ANOVA için grup değişkeninizde en az 2, tercihen 3 kategori olmalıdır. Sizde {len(unique_groups)} bulundu."}
        
        results = []
        for var in req.depVars:
            temp_df = df.dropna(subset=[var])
            
            group_stats = []
            group_data_list = []
            
            total_data = pd.to_numeric(temp_df[var], errors='coerce').dropna().values
            tot_n = len(total_data)
            if tot_n == 0: continue
            
            tot_mean = float(np.mean(total_data))
            tot_std = float(np.std(total_data, ddof=1)) if tot_n > 1 else 0.0
            
            for g in unique_groups:
                g_data = temp_df[temp_df[req.groupVar] == g][var]
                g_data = pd.to_numeric(g_data, errors='coerce').dropna().values
                group_data_list.append(g_data)
                
                n = len(g_data)
                mean = float(np.mean(g_data)) if n > 0 else 0.0
                std = float(np.std(g_data, ddof=1)) if n > 1 else 0.0
                group_stats.append({"val": str(g), "n": n, "mean": mean, "std": std})
            
            try:
                F_stat, p_val = stats.f_oneway(*group_data_list)
            except:
                F_stat, p_val = 0.0, 1.0
            
            post_hoc_pairs = []
            
            if not np.isnan(p_val) and p_val < 0.05:
                pairs = list(itertools.combinations(range(len(unique_groups)), 2))
                num_comparisons = len(pairs)
                
                for i, j in pairs:
                    if len(group_data_list[i]) < 2 or len(group_data_list[j]) < 2: continue
                    try:
                        t_stat, pair_p = stats.ttest_ind(group_data_list[i], group_data_list[j], equal_var=True)
                        adj_p = pair_p * num_comparisons if req.postHoc == 'Bonferroni' else pair_p
                        
                        if adj_p < 0.05:
                            if group_stats[i]["mean"] > group_stats[j]["mean"]:
                                post_hoc_pairs.append({"higher": str(unique_groups[i]), "lower": str(unique_groups[j])})
                            else:
                                post_hoc_pairs.append({"higher": str(unique_groups[j]), "lower": str(unique_groups[i])})
                    except:
                        pass
            
            results.append({
                "varName": var,
                "F": float(F_stat) if not np.isnan(F_stat) else 0.0,
                "p": float(p_val) if not np.isnan(p_val) else 1.0,
                "groups": group_stats,
                "total": {"n": tot_n, "mean": tot_mean, "std": tot_std},
                "postHocPairs": post_hoc_pairs
            })

        return {
            "groupVar": req.groupVar,
            "postHoc": req.postHoc,
            "originalGroups": [str(g) for g in unique_groups],
            "results": results
        }
    except Exception as e:
        return {"error": f"ANOVA Hatası: {str(e)}"}

@app.post("/normality")
def normality(req: NormalityRequest):
    try:
        df = pd.DataFrame(req.data)
        results = []
        
        for var in req.depVars:
            data = pd.to_numeric(df[var], errors='coerce').dropna().values
            n = len(data)
            if n < 3: continue
            
            mean = float(np.mean(data))
            median = float(np.median(data))
            
            vals, counts = np.unique(data, return_counts=True)
            mode = float(vals[np.argmax(counts)])
            
            std = float(np.std(data, ddof=1)) if n > 1 else 0.0
            
            if std == 0:
                results.append({
                    "varName": var, "n": n, "mean": mean, "median": median, "mode": mode,
                    "std": 0.0, "skewness": 0.0, "kurtosis": 0.0, "ks_stat": 0.0, "ks_p": 1.0
                })
                continue
            
            skew = float(stats.skew(data, bias=False))
            kurt = float(stats.kurtosis(data, bias=False))
            
            # SciPy "ndtr" argüman hatasını önlemek için veri matematiksel olarak standardize edilir (Z-Score)
            z_data = (data - mean) / std 
            
            # Artık args=(mean, std) parametresi göndermemize gerek kalmadan standart normal dağılımla sınanır
            ks_stat, ks_p = stats.kstest(z_data, 'norm')
            
            results.append({
                "varName": var,
                "n": n,
                "mean": mean,
                "median": median,
                "mode": mode,
                "std": std,
                "skewness": skew if not math.isnan(skew) else 0.0,
                "kurtosis": kurt if not math.isnan(kurt) else 0.0,
                "ks_stat": float(ks_stat) if not math.isnan(ks_stat) else 0.0,
                "ks_p": float(ks_p) if not math.isnan(ks_p) else 1.0
            })
            
        return {"results": results}
    except Exception as e:
        return {"error": f"Normallik Sınaması Hatası: {str(e)}"}
