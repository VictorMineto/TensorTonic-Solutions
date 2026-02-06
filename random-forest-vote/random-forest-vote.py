def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here

    predictions = list(predictions)
    if not predictions:
        return []
    
    num_samples = len(predictions[0])
    sample_predictions = []

    for i in range(num_samples):
        sample_preds = [tree_pred[i] for tree_pred in predictions]
        sample_predictions.append(sample_preds)

    final_predictions = []

    for sample_preds in sample_predictions:
        vote_counts = {}
        for pred in sample_preds:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1 
        
        max_votes = max(vote_counts.values())

        candidates = [cls for cls, count in vote_counts.items() if count == max_votes]
    
        final_predictions.append(min(candidates))
    
    return final_predictions