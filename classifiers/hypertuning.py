from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import make_scorer, f1_score
from classifiers.utils import get_stratified_kfold
from classifiers.classifier_init import all_classifiers
from preprocessing import all_scalers, sampler_classes
from utils.logger import log_memory_usage, setup_logger
import time

logger = setup_logger(__name__)

# Create a scorer object
f1_scorer = make_scorer(f1_score, average='binary')  

def hyperparameter_tuning(X_train, y_train, config, logger):
    tuned_parameters = config.get('tuned_parameters', {})
    selected_classifiers = config['classifiers']
    best_estimators = {}
    random_state = config.get('random_state', 25)
    resampling_config = config.get('resampling', {})
    resampling_method = resampling_config.get('method', None)
    resampling_params = resampling_config.get('params', {})
    scaler_name = config.get('scaling', None)
    cv_strategy = get_stratified_kfold(random_state)

    # Input validation for 'random_state'
    if random_state is None:
        raise ValueError("'random_state' is missing from the config. Please provide a valid integer value.")

    for name in selected_classifiers:
        logger.info(f"Processing classifier: {name}")

        steps = []
        if resampling_method:
            sampler_class = sampler_classes.get(resampling_method)
            if not sampler_class:
                raise ValueError(f"Invalid resampling method '{resampling_method}' in config.")
            steps.append(('resampling', sampler_class(**resampling_params)))

        if scaler_name:
            scaler_class = all_scalers.get(scaler_name)
            if not scaler_class:
                raise ValueError(f"Invalid scaler '{scaler_name}' in config.")
            steps.append(('scaler', scaler_class))

        steps.append(('classifier', all_classifiers[name]))
        pipeline = ImbPipeline(steps=steps)

        if name in tuned_parameters:
            # Hyperparameter tuning
            logger.info("Hyperparameter tuning for %s...", name)
            log_memory_usage(logger)  
            param_grid = {f'classifier__{key}': value for key, value in tuned_parameters[name].items()}

            grid_search = GridSearchCV(
                estimator=pipeline, 
                param_grid=param_grid, 
                scoring=f1_scorer, 
                cv=cv_strategy,
                n_jobs=-1
            )
            start_time = time.time()
            grid_search.fit(X_train, y_train)  
            end_time = time.time()
            logger.info(f"Hypertuning time for {name}: {end_time - start_time:.2f} seconds")
            log_memory_usage(logger)  
            best_estimators[name] = grid_search.best_estimator_
            logger.info("Best parameters for %s: %s", name, grid_search.best_params_)
        else:
            logger.warning(f"No tuning parameters specified for {name}. Using default settings.")
            best_estimators[name] = pipeline 

    return best_estimators
