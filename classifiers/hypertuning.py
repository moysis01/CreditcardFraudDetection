def hyperparameter_tuning(X_train, y_train, config, logger):
    from sklearn.model_selection import RandomizedSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.metrics import make_scorer, f1_score
    from classifiers.utils import calculate_n_iter, get_stratified_kfold
    from classifiers.classifier_init import all_classifiers
    from preprocessing import all_scalers, sampler_classes
    from utils.logger import log_memory_usage, setup_logger
    import time
    # Set up logger
    logger = setup_logger(__name__)

    # Create a scorer object
    f1_scorer = make_scorer(f1_score, average='binary')

    """
    Performs hyperparameter tuning for the specified classifiers using RandomizedSearchCV.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    config (dict): Configuration dictionary with tuning parameters.
    logger (Logger): Logger for logging information.

    Returns:
    dict: Best estimators for each classifier.
    """
    tuned_parameters = config.get('tuned_parameters', {})
    selected_classifiers = config['classifiers']
    best_estimators = {}
    resampling_methods = config.get('resampling', [])
    resampling_params = config.get('resampling_params', {})
    scaler_name = config.get('scaling', None)
    cv_strategy = get_stratified_kfold()  # Ensure this function is defined and imported

    for name in selected_classifiers:
        logger.info(f"Processing classifier: {name}")

        # Define the steps of the pipeline
        steps = []

        # Add resampling step if specified
        for method in resampling_methods:
            sampler_class = sampler_classes.get(method.upper())
            if not sampler_class:
                raise ValueError(f"Invalid resampling method '{method}' in config.")
            # Initialize the sampler with parameters if provided
            method_params = {key: resampling_params.get(key, [None]) for key in resampling_params if key in sampler_class.__init__.__code__.co_varnames}
            logger.info(f"Parameters for resampling method '{method}': {method_params}")
            sampler = sampler_class(**method_params)
            steps.append(('resampling', sampler))

        # Add scaler step if specified
        if scaler_name:
            scaler_class = all_scalers.get(scaler_name)
            if not scaler_class:
                raise ValueError(f"Invalid scaler '{scaler_name}' in config.")
            steps.append(('scaler', scaler_class))

        # Add classifier step
        steps.append(('classifier', all_classifiers[name]))
        pipeline = ImbPipeline(steps=steps)

        # Prepare parameter distributions for tuning
        param_distributions = {}
        if name in tuned_parameters:
            param_distributions.update({f'classifier__{key}': value for key, value in tuned_parameters[name].items()})
        
        if resampling_methods:
            for param_key, param_values in resampling_params.items():
                param_distributions.update({f'resampling__{param_key}': param_values})

        # Log parameter distributions for debugging
        logger.info("Parameter distributions for %s:", name)
        for param, values in param_distributions.items():
            logger.info(f"  {param}: {values}")

        if param_distributions:
            logger.info("Starting hyperparameter tuning for %s...", name)
            log_memory_usage(logger)  

            n_iter = calculate_n_iter(param_distributions, max_iter=100)  
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                scoring=f1_scorer,
                cv=cv_strategy,
                n_iter=n_iter,
                n_jobs=-1,
                verbose=1,
                random_state=25
            )
            try:
                start_time = time.time()
                random_search.fit(X_train, y_train)  
                end_time = time.time()
                logger.info(f"Hyperparameter tuning completed for {name}. Duration: {end_time - start_time:.2f} seconds")
                log_memory_usage(logger)
                best_estimators[name] = random_search.best_estimator_

                # Log the best parameters
                logger.info("Best parameters for %s:", name)
                best_params = random_search.best_params_
                for param, value in best_params.items():
                    logger.info(f"  {param}: {value}")
            except Exception as e:
                logger.error(f"Error during hyperparameter tuning for {name}: {e}", exc_info=True)
        else:
            logger.warning(f"No tuning parameters specified for {name}. Using default settings.")
            best_estimators[name] = pipeline 

    return best_estimators
