import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time_series_dataset import TimeSeriesDataset
from rnn_trainer import RNNModel

RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

class ImprovementLinks:

    """
        Improvement Links class to run experiments on filtered links.
        ---------------------------------
        This class will run experiments comparing stacking methods with baseline methods,
        focusing on filtered links to improve RMSE values and the improvement_pct column.
        After running the "prediction_all_links.py" file, the data will be filtered to include only five links,
        which are: sc-rn, df-go, ac-rn, am-rn, mt-ma.
    """

    def __init__(self, path_links="files_filtered"):
        self.path_links = Path(path_links)


    def setup_logger(self):
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("logs") / f"prediction_rf_best_{ts}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)7s | %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
        )
        return logging.getLogger(__name__), log_path


    def safe_load_csv(self, fp):
        try:
            return pd.read_csv(fp)
        except Exception:
            return pd.read_csv(fp, sep=";")


    def build_supervised(self,df, target_col="Atraso(ms)", lags=6):
        if "Data" in df.columns:
            df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
            df = df.sort_values("Data")
            df.set_index("Data", inplace=True)
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df[target_col].shift(lag)
        df[f"rolling_mean_{lags}"] = df[target_col].rolling(lags).mean().shift(1)
        for c in ["Hop_count", "Bottleneck", "Vazao_BBR", "mask_applied"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        y = df[target_col].astype(float)
        X = df.drop(columns=[target_col])
        return X, y


    def prepare_dl_data(self, df, target_col="Atraso(ms)"):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")
            df.set_index("Date", inplace=True)
        
        numeric_cols = [target_col]
        for c in ["Hop_count", "Bottleneck", "Vazao_BBR", "mask_applied"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                numeric_cols.append(c)
        
        data = df[numeric_cols].dropna()
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        return data_scaled, scaler


    def compute_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        eps = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100
        r2 = r2_score(y_true, y_pred)
        return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}




    def train_rf_cv(self, X, y, params, cv_splits=5):
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        metrics_accum = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            metrics_accum.append(self.compute_metrics(y_te.values, pred))
        agg = {k: float(np.mean([m[k] for m in metrics_accum])) for k in metrics_accum[0]}
        agg["params"] = params
        return agg


    def optimize_rf_params(self, X, y, param_ranges, cv_splits=5, n_trials=50):        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 
                                                param_ranges["n_estimators"][0], 
                                                param_ranges["n_estimators"][1]),
                "max_depth": trial.suggest_int("max_depth", 
                                            param_ranges["max_depth"][0], 
                                            param_ranges["max_depth"][1]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 
                                                    param_ranges["min_samples_leaf"][0], 
                                                    param_ranges["min_samples_leaf"][1])
            }
            
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            rmse_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                rmse = np.sqrt(mean_squared_error(y_te, pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        
        metrics = self.train_rf_cv(X, y, best_params, cv_splits)
        
        return best_params, metrics


    def train_dl_model(self, model, train_loader, val_loader, epochs, lr, patience=10):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//3, factor=0.5)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model


    def train_dl_cv(self, data_scaled, scaler, params, model_type, cv_splits=5):
        window = params["window"]
        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        dropout = params["dropout"]
        lr = params["lr"]
        epochs = params["epochs"]
        
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        metrics_accum = []
        
        for train_idx, test_idx in tscv.split(data_scaled):
            train_data = data_scaled[train_idx]
            test_data = data_scaled[test_idx]
            
            if len(train_data) <= window or len(test_data) <= window:
                continue
                
            train_dataset = TimeSeriesDataset(train_data, window)
            test_dataset = TimeSeriesDataset(test_data, window)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_size = data_scaled.shape[1]
            model = RNNModel(input_size, hidden_size, num_layers, model_type, dropout).to(DEVICE)
            
            model = self.train_dl_model(model, train_loader, test_loader, epochs, lr)
            
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(DEVICE)
                    outputs = model(X_batch)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(y_batch.numpy())
            
            if len(predictions) > 0:
                pred_array = np.array(predictions).reshape(-1, 1)
                actual_array = np.array(actuals).reshape(-1, 1)
                
                pred_full = np.zeros((len(predictions), data_scaled.shape[1]))
                actual_full = np.zeros((len(actuals), data_scaled.shape[1]))
                pred_full[:, 0] = predictions
                actual_full[:, 0] = actuals
                
                pred_denorm = scaler.inverse_transform(pred_full)[:, 0]
                actual_denorm = scaler.inverse_transform(actual_full)[:, 0]
                
                metrics_accum.append(self.compute_metrics(actual_denorm, pred_denorm))
        
        if not metrics_accum:
            return {"rmse": float('nan'), "mae": float('nan'), "mape": float('nan'), "r2": float('nan')}
        
        agg = {k: float(np.mean([m[k] for m in metrics_accum])) for k in metrics_accum[0]}
        agg["params"] = params
        return agg


    def optimize_dl_params(self, data_scaled, scaler, param_ranges, model_type, cv_splits=5, n_trials=30):
        
        def objective(trial):
            params = {
                "window": trial.suggest_int("window", param_ranges["window"][0], param_ranges["window"][1]),
                "hidden_size": trial.suggest_int("hidden_size", param_ranges["hidden_size"][0], param_ranges["hidden_size"][1]),
                "num_layers": trial.suggest_int("num_layers", param_ranges["num_layers"][0], param_ranges["num_layers"][1]),
                "dropout": trial.suggest_float("dropout", param_ranges["dropout"][0], param_ranges["dropout"][1]),
                "lr": trial.suggest_float("lr", param_ranges["lr"][0], param_ranges["lr"][1], log=True),
                "epochs": trial.suggest_int("epochs", param_ranges["epochs"][0], param_ranges["epochs"][1])
            }
            
            try:
                metrics = self.train_dl_cv(data_scaled, scaler, params, model_type, cv_splits)
                return metrics["rmse"] if not np.isnan(metrics["rmse"]) else float('inf')
            except Exception:
                return float('inf')
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        
        metrics = self.train_dl_cv(data_scaled, scaler, best_params, model_type, cv_splits)
        
        return best_params, metrics


    def run(self):
        logger, log_path = self.setup_logger()
        logger.info("Starting experiment with multiple models and links")

        '''
            Experiment list 
            ----------------------------------
            The parameters were chosen according to the results obtained after running the 558 links.
            Therefore, Optuna was used to perform the search within a range close to the parameters of the models analyzed
            in the output files "simple_comparison_summary.csv" and "simple_model_detail.csv", generated by the PredictionAllLinks class.
            The Random Forest values ​​were selected based on another experimental test with RF models on the same datasets.
        '''
        experiments = [
            {
                "link": "sc-rn",
                "model_type": "RandomForest",
                "files": [
                    ("baseline", "sc-rn_baseline_median.csv"),
                    ("stacking", "sc-rn_stacking.csv")
                ],
                "param_ranges": {
                    "baseline": {
                        "n_estimators": (120, 150),
                        "max_depth": (12, 16), 
                        "min_samples_leaf": (4, 6)
                    },
                    "stacking": {
                        "n_estimators": (100, 130),
                        "max_depth": (5, 8),
                        "min_samples_leaf": (4, 6)
                    }
                }
            },

            {
                "link": "df-go",
                "model_type": "GRU",
                "files": [
                    ("median", "df-go_baseline_median.csv"),
                    ("stacking", "df-go_stacking.csv")
                ],
                "param_ranges": {
                    "median": {
                        "window": (24, 48),
                        "hidden_size": (32, 48),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    },
                    "stacking": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    }
                }
            },
            {
                "link": "ac-rn",
                "model_type": "LSTM",
                "files": [
                    ("ffill", "ac-rn_baseline_ffill.csv"),
                    ("stacking", "ac-rn_stacking.csv")
                ],
                "param_ranges": {
                    "ffill": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    },
                    "stacking": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    }
                }
            },
            {
                "link": "am-rn",
                "model_type": "GRU",
                "files": [
                    ("ffill", "am-rn_baseline_ffill.csv"),
                    ("stacking", "am-rn_stacking.csv")
                ],
                "param_ranges": {
                    "ffill": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    },
                    "stacking": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    }
                }
            },

            {
                "link": "mt-ma",
                "model_type": "GRU",
                "files": [
                    ("ffill", "mt-ma_baseline_ffill.csv"),
                    ("stacking", "mt-ma_stacking.csv")
                ],
                "param_ranges": {
                    "ffill": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    },
                    "stacking": {
                        "window": (24, 48),
                        "hidden_size": (32, 64),
                        "num_layers": (2, 4),
                        "dropout": (0.05, 0.2),
                        "lr": (1e-4, 1e-2),
                        "epochs": (20, 35)
                    }
                }
            }
        ]

        all_detail_rows = []
        all_summary_rows = []

        for experiment in experiments:
            link = experiment["link"]
            model_type = experiment["model_type"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {link} with model {model_type}")
            logger.info(f"{'='*60}")
            
            detail_rows = []
            
            for variant, filename in experiment["files"]:
                file_path = self.path_links / filename
                
                if not file_path.exists():
                    logger.warning(f"File not found: {filename}")
                    continue
                    
                logger.info(f"\nOptimizing parameters for {variant}: {filename}")
                
                df = self.safe_load_csv(file_path)
                param_ranges = experiment["param_ranges"][variant]
                
                if model_type == "RandomForest":
                    X, y = self.build_supervised(df)
                    logger.info(f"RF ranges: {param_ranges}")
                    
                    best_params, metrics = self.optimize_rf_params(X, y, param_ranges, cv_splits=5, n_trials=50)
                    
                else: 
                    data_scaled, scaler = self.prepare_dl_data(df)
                    logger.info(f"DL ranges: {param_ranges}")
                    
                    if len(data_scaled) < param_ranges["window"][1] + 50:
                        logger.warning(f"Insufficient data for {variant}. Required > {param_ranges['window'][1] + 50} samples")
                        continue
                    
                    n_trials = 20 if model_type in ["GRU", "LSTM"] else 50
                    best_params, metrics = self.optimize_dl_params(data_scaled, scaler, param_ranges, model_type, cv_splits=5, n_trials=n_trials)
                
                logger.info(f"Best parameters found for {variant}: {best_params}")
                
                detail_row = {
                    "link": link,
                    "model_type": model_type,
                    "variant": variant,
                    "file": filename,
                    **metrics
                }
                detail_rows.append(detail_row)
                all_detail_rows.append(detail_row)
                
                logger.info(f"[{variant}] RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} R2={metrics['r2']:.4f}")

            if len(detail_rows) == 2:
                baseline_row = detail_rows[0]
                stacking_row = detail_rows[1] 
                
                improvement = (baseline_row["rmse"] - stacking_row["rmse"]) / baseline_row["rmse"] * 100.0
                
                summary_row = {
                    "link": link,
                    "model_type": model_type,
                    "baseline_file": baseline_row["file"],
                    "baseline_rmse": baseline_row["rmse"],
                    "stacking_file": stacking_row["file"],
                    "stacking_rmse": stacking_row["rmse"],
                    "improvement_pct": improvement
                }
                all_summary_rows.append(summary_row)
                
                logger.info(f"Stacking improvement: {improvement:.2f}%")

        df_detail = pd.DataFrame(all_detail_rows)
        df_detail.to_csv("all_models_detail.csv", index=False)
        logger.info(f"\nAll models details saved to all_models_detail.csv")

        if all_summary_rows:
            df_summary = pd.DataFrame(all_summary_rows)
            df_summary.to_csv("all_models_comparison_summary.csv", index=False)
            logger.info("Comparison summary saved to all_models_comparison_summary.csv")

        logger.info(f"\nComplete log: {log_path}")
        logger.info("End of complete experiment.")



