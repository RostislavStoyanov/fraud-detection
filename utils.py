import numpy as np

def get_noised_data(data):
    mean = 0
    var = 0.5
    noise = np.random.normal(mean, var, size=data.shape).astype(np.float32)
    return data + noise


def label_dist(data_df):
    """
    Print proportion of negative/positive samples
    :param data_df: dataframe containing values and class label('Class')
    :return:
    """
    val_cnts = data_df['Class'].value_counts()
    neg_samples = val_cnts[0]
    pos_samples = val_cnts[1]

    print("Non-fraud count:{:2d}, percentage:{:.4f}".format(neg_samples, neg_samples / len(data_df)))
    print("Fraud count:{:2d}, percentage:{:.4f}".format(pos_samples, pos_samples / len(data_df)))
