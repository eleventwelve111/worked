import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib


class DosePredictionModel:
    """
    Machine learning model for predicting dose rates based on simulation parameters.
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the dose prediction model.

        Parameters:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', or 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        self.trained = False

        # Initialize the selected model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def prepare_data(self, simulation_results):
        """
        Prepare data from simulation results for training.

        Parameters:
            simulation_results: Dictionary of simulation results

        Returns:
            pandas.DataFrame: Features dataframe
            pandas.Series: Target values
        """
        # Extract features and targets from simulation results
        features = []
        targets = []

        for config_key, results in simulation_results.items():
            # Parse configuration parameters
            energy_kev, channel_diameter = config_key.split('_')
            energy_kev = float(energy_kev)
            channel_diameter = float(channel_diameter)

            # Extract dose data at different positions
            dose_data = results['dose_data']

            for distance in dose_data:
                if distance == 'metadata':
                    continue

                for angle in dose_data[distance]:
                    if angle == 'spectrum':
                        continue

                    dose_info = dose_data[distance][angle]

                    # Extract dose value (prefer KERMA if available)
                    dose_value = None
                    if 'kerma' in dose_info:
                        dose_value = dose_info['kerma']
                    elif 'dose' in dose_info:
                        dose_value = dose_info['dose']
                    elif 'dose_equiv' in dose_info:
                        dose_value = dose_info['dose_equiv']

                    if dose_value is not None:
                        # Create feature vector
                        feature = {
                            'energy_kev': energy_kev,
                            'channel_diameter': channel_diameter,
                            'distance': float(distance),
                            'angle': float(angle)
                        }

                        # Add additional features if available
                        if 'material' in results:
                            feature['material_density'] = results['material'].get('density', 0.0)

                        if 'source' in results:
                            feature['source_strength'] = results['source'].get('strength', 0.0)

                        features.append(feature)
                        targets.append(dose_value)

        # Create dataframes
        features_df = pd.DataFrame(features)
        targets_series = pd.Series(targets, name='dose')

        self.feature_names = features_df.columns.tolist()
        self.target_name = 'dose'

        return features_df, targets_series

    def train(self, features_df, targets_series, hyperparameter_tuning=False):
        """
        Train the model on the prepared data.

        Parameters:
            features_df: Features dataframe
            targets_series: Target values
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            dict: Training results and model performance metrics
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_series, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == 'neural_network':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }

            # Perform grid search
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train_scaled, y_train)

            # Update model with best parameters
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_params = None

        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.trained = True

        # Evaluate model
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Calculate MAPE (Mean Absolute Percentage Error)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

        # Training results
        training_results = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'best_parameters': best_params,
            'metrics': {
                'train': {
                    'mse': float(train_mse),
                    'mae': float(train_mae),
                    'r2': float(train_r2),
                    'mape': float(train_mape)
                },
                'test': {
                    'mse': float(test_mse),
                    'mae': float(test_mae),
                    'r2': float(test_r2),
                    'mape': float(test_mape)
                }
            },
            'feature_importance': self._get_feature_importance()
        }

        return training_results

    def predict(self, new_features):
        """
        Make predictions with the trained model.

        Parameters:
            new_features: DataFrame or dictionary of features

        Returns:
            numpy.ndarray: Predicted dose values
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        # Convert to DataFrame if dictionary
        if isinstance(new_features, dict):
            new_features = pd.DataFrame([new_features])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in new_features.columns:
                raise ValueError(f"Missing feature: {feature}")

        # Scale features
        new_features_scaled = self.scaler.transform(new_features)

        # Make predictions
        predictions = self.model.predict(new_features_scaled)

        return predictions

    def save_model(self, filename):
        """
        Save the trained model to a file.

        Parameters:
            filename: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type
        }

        joblib.dump(model_data, filename)

    @classmethod
    def load_model(cls, filename):
        """
        Load a trained model from a file.

        Parameters:
            filename: Path to the saved model

        Returns:
            DosePredictionModel: Loaded model instance
        """
        model_data = joblib.load(filename)

        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.target_name = model_data['target_name']
        instance.trained = True

        return instance

    def _get_feature_importance(self):
        """
        Get feature importance from the trained model.

        Returns:
            dict: Feature importance scores
        """
        if not self.trained:
            return None

        feature_importance = {}

        if self.model_type in ['random_forest', 'gradient_boosting']:
            importances = self.model.feature_importances_
            for feature, importance in zip(self.feature_names, importances):
                feature_importance[feature] = float(importance)
        elif self.model_type == 'neural_network':
            # Neural networks don't have direct feature importance
            # Use permutation importance as an alternative
            from sklearn.inspection import permutation_importance

            # Create a small validation set from training data
            X = self.scaler.transform(pd.DataFrame(columns=self.feature_names))
            y = pd.Series([], name=self.target_name)

            if len(X) > 0:
                result = permutation_importance(
                    self.model, X, y, n_repeats=10, random_state=42
                )

                for feature, importance in zip(self.feature_names, result.importances_mean):
                    feature_importance[feature] = float(importance)
            else:
                # If no data available, return equal importance
                for feature in self.feature_names:
                    feature_importance[feature] = 1.0 / len(self.feature_names)

        return feature_importance

    def plot_model_performance(training_results, filename=None):
        """
        Plot model performance metrics and feature importance.

        Parameters:
            training_results: Dictionary of training results
            filename: Optional filename to save the plot

        Returns:
            matplotlib.figure.Figure: Figure with model performance plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Predicted vs Actual values
        ax = axes[0, 0]
        if 'predictions' in training_results:
            actual = training_results['predictions']['actual']
            predicted = training_results['predictions']['predicted']

            ax.scatter(actual, predicted, alpha=0.5)

            # Perfect prediction line
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            ax.set_xlabel('Actual Dose')
            ax.set_ylabel('Predicted Dose')
            ax.set_title('Predicted vs Actual Dose Values')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Prediction data not available', ha='center', va='center')

        # Plot 2: Feature Importance
        ax = axes[0, 1]
        if 'feature_importance' in training_results:
            feature_importance = training_results['feature_importance']

            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            features = [f[0] for f in sorted_features]
            importances = [f[1] for f in sorted_features]

            ax.barh(features, importances)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Feature importance not available', ha='center', va='center')

        # Plot 3: Error Metrics Comparison
        ax = axes[1, 0]
        if 'metrics' in training_results:
            metrics = training_results['metrics']

            # Extract metrics
            metric_names = ['mse', 'mae', 'mape']
            train_values = [metrics['train'][m] for m in metric_names]
            test_values = [metrics['test'][m] for m in metric_names]

            # Bar positions
            bar_width = 0.35
            index = np.arange(len(metric_names))

            ax.bar(index, train_values, bar_width, label='Training')
            ax.bar(index + bar_width, test_values, bar_width, label='Testing')

            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title('Model Error Metrics')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(['MSE', 'MAE', 'MAPE (%)'])
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Metrics not available', ha='center', va='center')

        # Plot 4: R² Score
        ax = axes[1, 1]
        if 'metrics' in training_results:
            metrics = training_results['metrics']

            train_r2 = metrics['train']['r2']
            test_r2 = metrics['test']['r2']

            ax.bar(['Training', 'Testing'], [train_r2, test_r2])
            ax.set_ylim(0, 1)
            ax.set_ylabel('R² Score')
            ax.set_title('Model Goodness of Fit (R²)')
            ax.grid(True, axis='y', alpha=0.3)

            # Add text labels
            ax.text(0, train_r2, f'{train_r2:.3f}', ha='center', va='bottom')
            ax.text(1, test_r2, f'{test_r2:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'R² scores not available', ha='center', va='center')

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        return fig