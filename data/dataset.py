# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import nibabel as nib
import os
from typing import Dict, List, Tuple, Optional, Any
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class NSCLCDataLoader:
    """Data loader for NSCLC radiogenomics datasets"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize scalers
        self.scalers = {
            'clinical': StandardScaler(),
            'radiomics_ct': StandardScaler(),
            'radiomics_pet': StandardScaler(),
            'pathways': StandardScaler()
        }
        
        # Label encoders for categorical variables
        self.label_encoders = {}
        
    def load_tcia_data(self) -> Dict[str, Any]:
        """Load TCIA NSCLC Radiogenomics dataset"""
        data_path = self.config.tcia_data_path
        
        # Load patient metadata
        metadata = pd.read_csv(os.path.join(data_path, 'clinical.csv'))
        
        # Load radiomic features
        ct_features = pd.read_csv(os.path.join(data_path, 'radiomics_ct.csv'))
        pet_features = pd.read_csv(os.path.join(data_path, 'radiomics_pet.csv'))
        
        # Load genomic data
        genomic_data = pd.read_csv(os.path.join(data_path, 'genomic.csv'))
        
        # Load pathway scores (pre-computed with GSVA)
        pathway_scores = pd.read_csv(os.path.join(data_path, 'pathway_scores.csv'))
        
        return {
            'metadata': metadata,
            'ct_features': ct_features,
            'pet_features': pet_features,
            'genomic_data': genomic_data,
            'pathway_scores': pathway_scores
        }
    
    def load_tcga_data(self) -> Dict[str, Any]:
        """Load TCGA-LUAD dataset for external validation"""
        data_path = self.config.tcga_data_path
        
        # Similar structure to TCIA but with LUAD-specific data
        metadata = pd.read_csv(os.path.join(data_path, 'clinical.csv'))
        ct_features = pd.read_csv(os.path.join(data_path, 'radiomics_ct.csv'))
        genomic_data = pd.read_csv(os.path.join(data_path, 'genomic.csv'))
        pathway_scores = pd.read_csv(os.path.join(data_path, 'pathway_scores.csv'))
        
        # Note: PET may not be available in TCGA-LUAD
        pet_features = None
        
        return {
            'metadata': metadata,
            'ct_features': ct_features,
            'pet_features': pet_features,
            'genomic_data': genomic_data,
            'pathway_scores': pathway_scores
        }
    
    def preprocess_clinical(self, clinical_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess clinical variables"""
        # Select relevant clinical features
        clinical_vars = [
            'age', 'gender', 'stage', 'histology', 'smoking_status',
            'pack_years', 'treatment', 'performance_status'
        ]
        
        # Handle missing values
        for var in clinical_vars:
            if var not in clinical_df.columns:
                clinical_df[var] = np.nan
        
        # Encode categorical variables
        categorical_vars = ['gender', 'stage', 'histology', 'smoking_status', 'treatment']
        for var in categorical_vars:
            if var in clinical_df.columns:
                if var not in self.label_encoders:
                    self.label_encoders[var] = LabelEncoder()
                    clinical_df[var] = clinical_df[var].fillna('Unknown')
                    clinical_df[var] = self.label_encoders[var].fit_transform(clinical_df[var].astype(str))
                else:
                    clinical_df[var] = clinical_df[var].fillna('Unknown')
                    clinical_df[var] = self.label_encoders[var].transform(clinical_df[var].astype(str))
        
        # Fill missing numerical values with median
        numerical_vars = ['age', 'pack_years', 'performance_status']
        for var in numerical_vars:
            if var in clinical_df.columns:
                clinical_df[var] = clinical_df[var].fillna(clinical_df[var].median())
        
        # Select and scale features
        feature_cols = [c for c in clinical_vars if c in clinical_df.columns]
        features = clinical_df[feature_cols].values
        
        # Fit or transform scaler
        if not hasattr(self.scalers['clinical'], 'mean_'):
            features_scaled = self.scalers['clinical'].fit_transform(features)
        else:
            features_scaled = self.scalers['clinical'].transform(features)
        
        return features_scaled, feature_cols
    
    def preprocess_radiomics(self, radiomics_df: pd.DataFrame, modality: str) -> np.ndarray:
        """Preprocess radiomic features"""
        # Select relevant radiomic features
        # Following IBSI guidelines, we select a comprehensive set
        
        # Remove non-feature columns
        if 'patient_id' in radiomics_df.columns:
            radiomics_df = radiomics_df.drop('patient_id', axis=1)
        
        # Fill missing values with median
        radiomics_df = radiomics_df.fillna(radiomics_df.median())
        
        # Scale features
        if not hasattr(self.scalers[f'radiomics_{modality}'], 'mean_'):
            features_scaled = self.scalers[f'radiomics_{modality}'].fit_transform(radiomics_df)
        else:
            features_scaled = self.scalers[f'radiomics_{modality}'].transform(radiomics_df)
        
        return features_scaled
    
    def preprocess_pathways(self, pathway_df: pd.DataFrame) -> np.ndarray:
        """Preprocess pathway scores from GSVA"""
        # Remove non-feature columns
        if 'patient_id' in pathway_df.columns:
            pathway_df = pathway_df.drop('patient_id', axis=1)
        
        # Fill missing values
        pathway_df = pathway_df.fillna(0)
        
        # Scale features
        if not hasattr(self.scalers['pathways'], 'mean_'):
            features_scaled = self.scalers['pathways'].fit_transform(pathway_df)
        else:
            features_scaled = self.scalers['pathways'].transform(pathway_df)
        
        return features_scaled
    
    def create_dataset(self, data_type: str = 'tcia', split: str = 'train') -> 'NSCLCDataset':
        """Create PyTorch dataset"""
        if data_type == 'tcia':
            raw_data = self.load_tcia_data()
        elif data_type == 'tcga':
            raw_data = self.load_tcga_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Preprocess each modality
        clinical_features, clinical_cols = self.preprocess_clinical(raw_data['metadata'])
        ct_features = self.preprocess_radiomics(raw_data['ct_features'], 'ct')
        
        if raw_data.get('pet_features') is not None:
            pet_features = self.preprocess_radiomics(raw_data['pet_features'], 'pet')
        else:
            pet_features = None
        
        pathway_scores = self.preprocess_pathways(raw_data['pathway_scores'])
        
        # Extract survival and recurrence labels
        survival_times = raw_data['metadata']['survival_time'].values
        event_indicators = raw_data['metadata']['event'].values
        recurrence_labels = raw_data['metadata'].get('recurrence_3yr', np.zeros(len(survival_times)))
        
        # Get patient IDs
        patient_ids = raw_data['metadata']['patient_id'].values.tolist()
        
        return NSCLCDataset(
            patient_ids=patient_ids,
            clinical_features=clinical_features,
            ct_features=ct_features,
            pet_features=pet_features,
            pathway_scores=pathway_scores,
            survival_times=survival_times,
            event_indicators=event_indicators,
            recurrence_labels=recurrence_labels,
            clinical_feature_names=clinical_cols
        )


class NSCLCDataset(Dataset):
    """PyTorch dataset for NSCLC radiogenomics data"""
    
    def __init__(self, patient_ids, clinical_features, ct_features, pet_features,
                 pathway_scores, survival_times, event_indicators,
                 recurrence_labels=None, clinical_feature_names=None):
        
        self.patient_ids = patient_ids
        self.clinical_features = torch.FloatTensor(clinical_features)
        self.ct_features = torch.FloatTensor(ct_features)
        self.pet_features = torch.FloatTensor(pet_features) if pet_features is not None else None
        self.pathway_scores = torch.FloatTensor(pathway_scores)
        self.survival_times = torch.FloatTensor(survival_times)
        self.event_indicators = torch.FloatTensor(event_indicators)
        self.recurrence_labels = torch.FloatTensor(recurrence_labels) if recurrence_labels is not None else None
        self.clinical_feature_names = clinical_feature_names
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        item = {
            'patient_id': self.patient_ids[idx],
            'modality_data': {},
            'clinical_features': self.clinical_features[idx],
            'ct_features': self.ct_features[idx],
            'pathway_scores': self.pathway_scores[idx],
            'survival_time': self.survival_times[idx],
            'event_indicator': self.event_indicators[idx]
        }
        
        if self.pet_features is not None:
            item['modality_data']['PET'] = self.pet_features[idx].unsqueeze(0)  # Add channel dim
        
        if self.recurrence_labels is not None:
            item['recurrence_label'] = self.recurrence_labels[idx]
        
        # Add modality-specific data
        item['modality_data']['CT'] = item['ct_features'].unsqueeze(0)  # Add channel dim
        item['modality_data']['Clinical'] = item['clinical_features']
        item['modality_data']['Genomic'] = item['pathway_scores']
        
        return item
    
    def get_patient_graphs(self, graph_constructor, cohort_correlations):
        """Pre-compute patient graphs for the entire dataset"""
        patient_graphs = []
        
        for i in range(len(self)):
            patient_data = {
                'patient_id': self.patient_ids[i],
                'CT_features': self.ct_features[i].numpy(),
                'PET_features': self.pet_features[i].numpy() if self.pet_features is not None else None,
                'clinical_features': self.clinical_features[i].numpy(),
                'pathway_scores': self.pathway_scores[i].numpy()
            }
            
            graph = graph_constructor.build_graph(patient_data, cohort_correlations)
            patient_graphs.append(graph)
        
        return patient_graphs
