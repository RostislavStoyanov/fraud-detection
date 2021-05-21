import numpy as np
import loras


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


def get_x_y(df):
    y = df['Class'].astype(np.float32)
    X = df.drop('Class', axis=1).astype(np.float32)

    return X, y


def loras_oversample_dataframe(df):
    min_class = df.loc[df['Class'] == 1].drop('Class', axis=1).values
    maj_class = df.loc[df['Class'] == 0].drop('Class', axis=1).values
    k = 30
    num_shadow_points = 100
    sigma = [.005] * min_class.shape[1]
    num_generated_points = (len(maj_class) - len(min_class)) // len(min_class)
    num_aff_comb = 300
    seed = 42
    new_min_class = loras.fit_resample(maj_class, min_class, k=k, num_shadow_points=num_shadow_points,
                                       random_state=seed, list_sigma_f=sigma, num_generated_points=num_generated_points,
                                       num_aff_comb=num_aff_comb, )

    new_X = np.concatenate((maj_class, new_min_class))
    new_y = np.concatenate((np.zeros(len(maj_class)), np.ones(len(new_min_class))))

    return new_X.astype(np.float32), new_y.astype(np.float32)
