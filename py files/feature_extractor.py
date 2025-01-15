import pandas as pd
import numpy as np
from Bio.SeqUtils import gc_fraction
from sklearn.preprocessing import StandardScaler

class CRISPRFeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    class CRISPRFeatureExtractor:
        def __init__(self, sequence):
            self.sequence = sequence.upper()
            self.guide = self.sequence[:-3]  # 20nt guide sequence
            self.pam = self.sequence[-3:]    # 3bp PAM
            
            # Define all possible dinucleotides
            self.all_dinucs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 
                              'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
            
            # Cut site is typically between positions 17 and 18 of guide
            self.cut_site_index = len(self.guide) - 3
            
        def get_position_features(self):
            """One-hot encode guide sequence and PAM"""
            nucleotides = ['A', 'T', 'C', 'G']
            features = {}
            
            # One-hot encode guide sequence
            for i, nt in enumerate(self.guide):
                for n in nucleotides:
                    features[f'guide_pos_{i}_{n}'] = 1 if nt == n else 0
                    
            # One-hot encode PAM
            for i, nt in enumerate(self.pam):
                for n in nucleotides:
                    features[f'pam_pos_{i}_{n}'] = 1 if nt == n else 0
                    
            return features
        
        def get_cut_site_context(self):
            """Get features around cut site"""
            features = {}
            cut_start = max(0, self.cut_site_index - 3)
            cut_end = min(len(self.guide), self.cut_site_index + 4)
            cut_context = self.guide[cut_start:cut_end]
            
            nucleotides = ['A', 'T', 'C', 'G']
            for i, nt in enumerate(cut_context):
                pos = i - 3  # Position relative to cut site
                for n in nucleotides:
                    features[f'cut_site_{pos}_{n}'] = 1 if nt == n else 0
                    
            return features
        
        def get_sequence_features(self):
            """Get sequence features"""
            features = {}
            
            features['gc_content'] = gc_fraction(self.guide)
            
            for dinuc in self.all_dinucs:
                features[f'dinuc_{dinuc}'] = 0
                
            for i in range(len(self.guide)-1):
                dinuc = self.guide[i:i+2]
                features[f'dinuc_{dinuc}'] += 1
            
            return features
        
        def get_all_features(self):
            """Combine all features"""
            features = {}
            features.update(self.get_sequence_features())
            features.update(self.get_position_features())
            features.update(self.get_cut_site_context())
            return features
    
    def extract_features(self, df, fit_scaler=True):
        """Extract and scale features."""
        features_list = []
        ids = []
        
        for idx, row in df.iterrows():
            extractor = self.CRISPRFeatureExtractor(row['GuideSeq'])
            features = extractor.get_all_features()
            features_list.append(features)
            ids.append(row['Id'])
        
        feature_df = pd.DataFrame(features_list)
        feature_df['Id'] = ids
        
        if self.feature_columns is None:
            self.feature_columns = [col for col in feature_df.columns if col != 'Id']
        
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(feature_df[self.feature_columns])
        else:
            scaled_features = self.scaler.transform(feature_df[self.feature_columns])
            
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_columns)
        scaled_df['Id'] = ids
        
        return scaled_df
    
    def process_data(self, df, is_training=True):
        """Process data for training or prediction."""
        features_df = self.extract_features(df, fit_scaler=is_training)
        
        if is_training:
            targets = ['Fraction_Insertions', 'Avg_Deletion_Length',
                      'Indel_Diversity', 'Fraction_Frameshifts']
            result_df = pd.merge(features_df, 
                               df[['Id'] + targets],
                               on='Id')
        else:
            result_df = features_df
            
        print(f"Processed {len(result_df)} sequences")
        print(f"Total features: {len(self.feature_columns)}")
        print(f"Missing values: {result_df.isnull().sum().sum()}")
        
        return result_df