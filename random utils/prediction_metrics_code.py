import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve, precision_score, average_precision_score
from sklearn.dummy import DummyClassifier

def get_prob_for_precision(precisions, thresholds, precision_required, tolerance = 0.02):

    #use if you want to tweak the precision level using the prob thresholds and see results
    for i in range(len(precisions)):
        if (precision_required - tolerance) <= precisions[i] <= (precision_required + tolerance):
            return thresholds[i]
        
    #print('no threshold found for given precision and tolerance')
    return None

def get_prob_for_recall(recalls, thresholds, recall_required, tolerance = 0.02):
    #use if you want to tweak the recall level using the prob thresholds and see results
    for i in range(len(recalls)):
        if (recall_required - tolerance) <= recalls[i] <= (recall_required + tolerance):
            return thresholds[i]
        
    #print('no threshold found for given recall and tolerance')
    return None    

def binary_performances(y_true, y_prob, thresh=0.5, labels=['Positives', 'Negatives']):
    
    shape = y_prob.shape
    if len(shape) > 1:
        if shape[1] > 2:
            raise ValueError('A binary class problem is required')
        else:
            y_prob = y_prob[:, 1]

    weighted_precision = precision_score(y_true, (y_prob > thresh).astype(int), average = 'weighted')
    avg_precision = average_precision_score(y_true, (y_prob > thresh).astype(int))

    #added dummy classifier for comparison
    # no skill model, stratified random class predictions
    dummy_x = np.arange(len(y_true))
    dummy_model = DummyClassifier(strategy='stratified')
    dummy_model.fit(dummy_x, y_true) 
    dummy_yhat = dummy_model.predict_proba(dummy_x)
    dummy_pos_probs = dummy_yhat[:, 1]    

    plt.figure(figsize=[16, 16])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y_true, (y_prob > thresh).astype(int))

    plt.subplot(3,2,1, aspect='equal', adjustable='box')
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)
    plt.subplots_adjust(wspace=.25, hspace=.25)  

    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # 2 -- Distributions of Predicted Probabilities of both classes
    plt.subplot(3,2,2, adjustable='box')
    plt.hist(y_prob[y_true == 1], density=True, bins=25, alpha=0.5, color='green', label=labels[0])
    plt.hist(y_prob[y_true == 0], density=True, bins=25, alpha=0.5, color='red', label=labels[1])
    plt.axvline(thresh, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0, 1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=.25, hspace=.25)  

    #3 -- ROC curve with annotated decision point
    dummy_fp_rates, dummy_tp_rates, _ = roc_curve(y_true, dummy_pos_probs)
    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    dummy_roc_auc = auc(dummy_fp_rates, dummy_tp_rates)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(3,2,3, aspect='equal', adjustable='box')
    plt.plot(fp_rates, tp_rates, color='orange',
             lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot(dummy_fp_rates, dummy_tp_rates, color='red',
             lw=1, label='Dummy ROC curve (area = %0.3f)' % dummy_roc_auc)             
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=.25, hspace=.25)  

    # 4 -- Precision-Recall curve
    plt.subplot(3,2,4, aspect='equal', adjustable='box')
    dummy_precisions, dummy_recalls, dummy_thresholds = precision_recall_curve(y_true, dummy_pos_probs)
    dummy_pr_roc_auc = auc(dummy_recalls, dummy_precisions)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_roc_auc = auc(recalls, precisions)
    plt.plot(recalls, precisions, color='purple', lw=1, label='Precision-Recall curve')
    plt.plot(dummy_precisions, dummy_recalls, color='green', lw=1, label='Dummy Precision-Recall curve')
    plt.plot(recall, precision, 'bo', markersize=8, label='Decision Point')
    plt.xlabel('Recall', size=13)
    plt.ylabel('Precision', size=13)
    plt.title('Precision-Recall Curve', size=15)
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=.25, hspace=.25)  


    # 5 -- Model Lift Chart
    plt.subplot(3,2,5, aspect='equal', adjustable='box')   
    # Sort predictions and actual values by predicted probability
    y_true_temp = np.array(y_true.to_list())
    sorted_indices = np.argsort(y_prob)
    y_hat_sorted = y_prob[sorted_indices]
    y_target_sorted = y_true_temp[sorted_indices][::-1]

    dummy_sorted_indices = np.argsort(dummy_pos_probs)
    dummy_y_hat_sorted = dummy_pos_probs[dummy_sorted_indices]
    dummy_y_target_sorted = y_true_temp[dummy_sorted_indices][::-1]

    num_samples = len(y_true_temp)
    num_conversions = sum(y_true_temp)

    # Calculate cumulative percentage of conversions and samples mailed
    cumulative_conversions = np.cumsum(y_target_sorted) / num_conversions * 100
    dummy_cumulative_conversions = np.cumsum(dummy_y_target_sorted) / num_conversions * 100
    cumulative_mailed = np.linspace(0, 100, num_samples)   
    plt.plot(cumulative_mailed, cumulative_conversions, marker='o', label = 'Model Results')
    plt.plot(cumulative_mailed, dummy_cumulative_conversions, marker='o', label = 'Dummy Results')
    plt.xlabel('% Treated', size=13)
    plt.ylabel('% Captured', size=13)
    plt.title('Cumulative Lift Chart', size=15)
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=.25, hspace=.25)  

    # 6 -- Population Coverage at Specific Recall Levels
    #this enables you to understand better the precision-recall tradeoffs present.....
    plt.subplot(3,2,6, aspect='equal', adjustable='box')
    recall_step = recall*.05
    objective_recalls = [round(recall + (recall_step * i),2) for i in range(-5,6) if i != 0]

    prob_items = []
    for recall_item in objective_recalls:
        prob_items.append(get_prob_for_recall(recalls, thresholds, recall_required = recall_item))

    plt.plot(recalls, precisions, color='purple', lw=1, label='Precision-Recall curve')
    plt.plot(recall, precision, 'bo', markersize=8, label='Decision Point')

    for i in range(len(prob_items)):
        if prob_items[i] is None:
            continue
        elif prob_items[i] == thresh:
            continue
        else:
            thresh_temp = prob_items[i]
            label = f'Recall {objective_recalls[i]}'
            count_selected = (y_prob >= thresh_temp).sum()
            label += f' count: {count_selected}'
            cm_temp = confusion_matrix(y_true, (y_prob > thresh_temp).astype(int))
            tn_temp, fp_temp, fn_temp, tp_temp = [i for i in cm_temp.ravel()]
            precision_temp = tp_temp / (tp_temp + fp_temp)
            recall_temp = tp_temp / (tp_temp + fn_temp) 
            plt.plot(recall_temp, precision_temp, marker = "o", markersize=8, label=label)

    plt.xlabel('Recall', size=13)
    plt.ylabel('Precision', size=13)
    plt.title('Precision-Recall Curve Adjustments', size=15)
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=.25, hspace=.25)   

    plt.show()


    F1 = 2*(precision * recall) / (precision + recall)
    results = {
        "Prob Threshold": thresh,
        "Precision": precision, 
        "Weighted Precision": weighted_precision,
        "Average Precision" : avg_precision,
        "Recall": recall,
        "F1 Score": F1, 
        "AUC": roc_auc, 
        "Dummy AUC": dummy_roc_auc,
        "PR-AUC":pr_roc_auc,
        "Dummy PR-AUC": dummy_pr_roc_auc
    }
    
    prints = [f"{kpi}: {round(score, 3)}" for kpi,score in results.items()]
    prints = ' | '.join(prints)
    print(prints)

    return roc_auc, precisions, thresholds   

