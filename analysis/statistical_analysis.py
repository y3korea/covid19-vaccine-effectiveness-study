#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-19 Vaccine Effectiveness Meta-Analysis
Author: Wansuk Choi
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

class VaccineMetaAnalysis:
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load meta-analysis data"""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        return self.data
    
    def calculate_pooled_effect(self, effects, weights=None, method='random'):
        """Calculate pooled effect size"""
        if weights is None:
            weights = np.ones(len(effects))
        
        if method == 'fixed':
            pooled = np.average(effects, weights=weights)
            se = np.sqrt(1 / np.sum(weights))
        else:  # random effects
            # DerSimonian-Laird method
            Q = np.sum(weights * (effects - np.average(effects, weights=weights))**2)
            df = len(effects) - 1
            tau2 = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
            
            weights_random = 1 / (1/weights + tau2)
            pooled = np.average(effects, weights=weights_random)
            se = np.sqrt(1 / np.sum(weights_random))
        
        ci_lower = pooled - 1.96 * se
        ci_upper = pooled + 1.96 * se
        
        return pooled, ci_lower, ci_upper, se
    
    def calculate_heterogeneity(self, effects, weights):
        """Calculate heterogeneity statistics"""
        pooled = np.average(effects, weights=weights)
        Q = np.sum(weights * (effects - pooled)**2)
        df = len(effects) - 1
        p_value = 1 - stats.chi2.cdf(Q, df)
        
        # I-squared
        I2 = max(0, (Q - df) / Q * 100)
        
        # Tau-squared (DerSimonian-Laird)
        tau2 = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
        
        return {
            'Q': Q,
            'df': df,
            'p_value': p_value,
            'I2': I2,
            'tau2': tau2
        }
    
    def create_forest_plot(self, studies, effects, ci_lower, ci_upper, weights=None):
        """Create forest plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        n_studies = len(studies)
        y_pos = np.arange(n_studies)
        
        # Plot individual studies
        for i in range(n_studies):
            ax.plot([ci_lower[i], ci_upper[i]], [i, i], 'k-', linewidth=1)
            size = weights[i] * 100 if weights is not None else 50
            ax.scatter(effects[i], i, s=size, c='blue', alpha=0.6)
        
        # Add pooled effect
        pooled, pooled_lower, pooled_upper, _ = self.calculate_pooled_effect(effects, weights)
        ax.axvline(x=pooled, color='red', linestyle='--', label='Pooled Effect')
        ax.fill_betweenx([-1, n_studies], pooled_lower, pooled_upper, 
                         alpha=0.2, color='red')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(studies)
        ax.set_xlabel('Vaccine Effectiveness (%)', fontsize=12)
        ax.set_title('Forest Plot: COVID-19 Vaccine Effectiveness', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def subgroup_analysis(self, groups, effects, weights):
        """Perform subgroup analysis"""
        unique_groups = np.unique(groups)
        subgroup_results = {}
        
        for group in unique_groups:
            mask = groups == group
            group_effects = effects[mask]
            group_weights = weights[mask] if weights is not None else None
            
            pooled, ci_lower, ci_upper, se = self.calculate_pooled_effect(
                group_effects, group_weights
            )
            heterogeneity = self.calculate_heterogeneity(group_effects, group_weights)
            
            subgroup_results[group] = {
                'n_studies': len(group_effects),
                'pooled_effect': pooled,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'heterogeneity': heterogeneity
            }
        
        return subgroup_results
    
    def sensitivity_analysis(self, effects, weights, study_names):
        """Leave-one-out sensitivity analysis"""
        sensitivity_results = []
        
        for i in range(len(effects)):
            # Remove one study
            mask = np.ones(len(effects), dtype=bool)
            mask[i] = False
            
            effects_loo = effects[mask]
            weights_loo = weights[mask] if weights is not None else None
            
            pooled, ci_lower, ci_upper, _ = self.calculate_pooled_effect(
                effects_loo, weights_loo
            )
            
            sensitivity_results.append({
                'excluded_study': study_names[i],
                'pooled_effect': pooled,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
        
        return pd.DataFrame(sensitivity_results)
    
    def run_analysis(self):
        """Run complete meta-analysis"""
        # Load data
        self.load_data()
        
        # Example analysis with dummy data
        studies = ['Study ' + str(i+1) for i in range(10)]
        effects = np.random.normal(75, 10, 10)  # Vaccine effectiveness
        weights = np.random.uniform(0.5, 2, 10)  # Study weights
        
        # Overall pooled effect
        pooled, ci_lower, ci_upper, se = self.calculate_pooled_effect(effects, weights)
        print(f"Pooled Vaccine Effectiveness: {pooled:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%)")
        
        # Heterogeneity
        heterogeneity = self.calculate_heterogeneity(effects, weights)
        print(f"Heterogeneity: I² = {heterogeneity['I2']:.1f}%, p = {heterogeneity['p_value']:.3f}")
        
        # Create forest plot
        ci_lower_studies = effects - 1.96 * np.random.uniform(2, 5, 10)
        ci_upper_studies = effects + 1.96 * np.random.uniform(2, 5, 10)
        fig = self.create_forest_plot(studies, effects, ci_lower_studies, ci_upper_studies, weights)
        fig.savefig('../results/forest_plot.png', dpi=300, bbox_inches='tight')
        
        return self.results

if __name__ == "__main__":
    # Run analysis
    analysis = VaccineMetaAnalysis('../data/meta_analysis_results.json')
    results = analysis.run_analysis()
    print("\nMeta-analysis completed successfully!")
