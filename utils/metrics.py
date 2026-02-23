# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc, confusion_matrix, calibration_curve
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional


class ModelEvaluator:
    """Evaluate MS-HHGN model performance"""
    
    @staticmethod
    def compute_c_index(survival_times: np.ndarray, predicted_risk: np.ndarray, 
                        events: np.ndarray) -> float:
        """Compute concordance index"""
        from lifelines.utils import concordance_index
        return concordance_index(survival_times, -predicted_risk, events)
    
    @staticmethod
    def compute_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
        """Compute area under ROC curve"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, predictions)
    
    @staticmethod
    def kaplan_meier_analysis(survival_times: np.ndarray, events: np.ndarray,
                              risk_scores: np.ndarray, threshold: Optional[float] = None):
        """Perform Kaplan-Meier analysis with risk stratification"""
        if threshold is None:
            threshold = np.median(risk_scores)
        
        high_risk = risk_scores > threshold
        low_risk = ~high_risk
        
        # Fit Kaplan-Meier curves
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        kmf_high.fit(survival_times[high_risk], events[high_risk], label='High Risk')
        kmf_low.fit(survival_times[low_risk], events[low_risk], label='Low Risk')
        
        # Log-rank test
        results = logrank_test(
            survival_times[high_risk], survival_times[low_risk],
            event_observed_A=events[high_risk], event_observed_B=events[low_risk]
        )
        
        # Compute hazard ratio
        from lifelines import CoxPHFitter
        df = pd.DataFrame({
            'time': survival_times,
            'event': events,
            'risk_group': (risk_scores > threshold).astype(int)
        })
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='event')
        hr = np.exp(cph.params_.iloc[0])
        hr_ci = np.exp(cph.confidence_intervals_.iloc[0].values)
        
        return {
            'kmf_high': kmf_high,
            'kmf_low': kmf_low,
            'logrank_p': results.p_value,
            'hazard_ratio': hr,
            'hr_ci_lower': hr_ci[0],
            'hr_ci_upper': hr_ci[1],
            'threshold': threshold
        }
    
    @staticmethod
    def calibration_curve(labels: np.ndarray, predictions: np.ndarray, 
                          n_bins: int = 10) -> Dict:
        """Compute calibration curve and metrics"""
        prob_true, prob_pred = calibration_curve(labels, predictions, n_bins=n_bins)
        
        # Expected calibration error
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        # Brier score
        brier = np.mean((predictions - labels) ** 2)
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece,
            'brier': brier
        }
    
    @staticmethod
    def decision_curve_analysis(labels: np.ndarray, predictions: np.ndarray,
                                thresholds: np.ndarray) -> Dict:
        """Perform decision curve analysis"""
        n = len(labels)
        net_benefit = []
        
        for t in thresholds:
            # Treat all
            nb_all = (labels.mean() - t * (1 - labels.mean()))
            
            # Treat none
            nb_none = 0
            
            # Model-based
            decisions = (predictions >= t).astype(int)
            tp = np.sum((decisions == 1) & (labels == 1))
            fp = np.sum((decisions == 1) & (labels == 0))
            nb_model = (tp / n) - (fp / n) * (t / (1 - t))
            
            net_benefit.append({
                'threshold': t,
                'model': nb_model,
                'treat_all': nb_all,
                'treat_none': nb_none
            })
        
        return net_benefit
    
    @staticmethod
    def plot_roc_curve(labels: np.ndarray, predictions: np.ndarray, 
                       ax=None, label='MS-HHGN', color='blue'):
        """Plot ROC curve"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        return ax
    
    @staticmethod
    def plot_calibration(labels: np.ndarray, predictions: np.ndarray,
                         ax=None, label='MS-HHGN', color='blue'):
        """Plot calibration curve"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        prob_true, prob_pred = calibration_curve(labels, predictions, n_bins=10)
        
        ax.plot(prob_pred, prob_true, marker='o', color=color, label=label)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        
        return ax


class InterpretabilityAnalyzer:
    """Analyze model interpretability outputs"""
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray,
                                uncertainties: Optional[np.ndarray] = None,
                                top_k: int = 15, ax=None):
        """Plot feature importance with confidence intervals"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by importance
        idx = np.argsort(importance_scores)[::-1][:top_k]
        
        y_pos = np.arange(len(idx))
        
        if uncertainties is not None:
            ax.barh(y_pos, importance_scores[idx], xerr=1.96 * uncertainties[idx],
                   align='center', alpha=0.8, capsize=3)
        else:
            ax.barh(y_pos, importance_scores[idx], align='center', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Feature Importance')
        
        return ax
    
    @staticmethod
    def plot_attention_heatmap(attention_weights: Dict[str, np.ndarray],
                               query_modalities: List[str],
                               key_modalities: List[str],
                               ax=None):
        """Plot cross-modal attention heatmap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create attention matrix
        attn_matrix = np.zeros((len(query_modalities), len(key_modalities)))
        
        for i, q in enumerate(query_modalities):
            for j, k in enumerate(key_modalities):
                key = f"{q}→{k}"
                if key in attention_weights:
                    attn_matrix[i, j] = attention_weights[key].mean()
        
        sns.heatmap(attn_matrix, annot=True, fmt='.3f', 
                   xticklabels=key_modalities, yticklabels=query_modalities,
                   cmap='YlOrRd', ax=ax)
        ax.set_xlabel('Key Modalities')
        ax.set_ylabel('Query Modalities')
        ax.set_title('Cross-Modal Attention Weights')
        
        return ax
    
    @staticmethod
    def plot_meta_path_weights(meta_path_weights: np.ndarray,
                              meta_path_names: List[str],
                              ax=None):
        """Plot semantic attention weights for meta-paths"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Box plot of weights across patients
        bp = ax.boxplot(meta_path_weights, labels=meta_path_names, patch_artist=True)
        
        # Customize colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Attention Weight')
        ax.set_title('Meta-Path Importance Weights')
        ax.grid(True, alpha=0.3)
        
        return ax
