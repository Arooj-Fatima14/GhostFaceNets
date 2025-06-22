#!/usr/bin/env python3

import pickle
import os
import io
from tqdm import tqdm
from skimage.io import imread
from sklearn.preprocessing import normalize
import tensorflow as tf
import numpy as np
import glob2
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.decomposition import PCA

class eval_callback(tf.keras.callbacks.Callback):
    def __init__(self, basic_model, test_bin_file, batch_size=128, save_model=None, eval_freq=1, flip=True, PCA_acc=False):
        super(eval_callback, self).__init__()
        # Load the pickled data directly
        with open(test_bin_file, 'rb') as f:
            bins, issame_list = pickle.load(f)
        
        # Convert bins to a NumPy array of byte strings
        bins = np.array(bins, dtype=object)
        if bins.ndim != 1 or bins.dtype != object:
            raise ValueError(f"Expected 1D array of byte strings for bins, got shape {bins.shape} and dtype {bins.dtype}")
        
        # Filter and validate bins (ensure all are bytes or bytearrays)
        valid_bins = []
        for i, bin_data in enumerate(bins):
            if isinstance(bin_data, (bytes, bytearray)):
                valid_bins.append(bin_data)
            else:
                print(f"Skipping invalid bin data at index {i}: {type(bin_data)}")
        bins = np.array(valid_bins, dtype=object)
        if len(bins) == 0:
            raise ValueError("No valid byte data found in bins")
        
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices(bins)
        _imread = lambda xx: (tf.cast(tf.image.decode_image(xx, channels=3), "float32") - 127.5) * 0.0078125
        ds = ds.map(_imread, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        self.ds = ds.batch(batch_size)
        
        # Convert issame_list to NumPy array
        self.test_issame = np.array(issame_list, dtype=bool)
        self.test_names = os.path.splitext(os.path.basename(test_bin_file))[0]
        self.steps = int(np.ceil(len(bins) / batch_size))
        self.basic_model = basic_model
        self.max_accuracy, self.cur_acc, self.acc_thresh = 0.0, 0.0, 0.0
        self.save_model, self.eval_freq, self.flip, self.PCA_acc = save_model, eval_freq, flip, PCA_acc
        if eval_freq > 1:
            self.on_batch_end = lambda batch=0, logs=None: self.__eval_func__(batch, logs, eval_freq=eval_freq)
        self.on_epoch_end = lambda epoch=0, logs=None: self.__eval_func__(epoch, logs, eval_freq=1)

        self.is_distribute = False
        if tf.distribute.has_strategy():
            self.strategy = tf.distribute.get_strategy()
            self.num_replicas = self.strategy.num_replicas_in_sync
            if self.num_replicas > 1:
                self.is_distribute = True
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
                self.ds = self.strategy.experimental_distribute_dataset(self.ds.with_options(options))

    def __do_predict__(self):
        embs = []
        for img_batch in tqdm(self.ds, "Evaluating " + self.test_names, total=self.steps):
            emb = self.basic_model(img_batch, training=False)
            if self.flip:
                emb_f = self.basic_model(tf.image.flip_left_right(img_batch), training=False)
                emb = emb + emb_f
            embs.append(emb.numpy())
        return np.concatenate(embs, axis=0)

    def __do_predict_distribute__(self):
        embs = []
        for img_batch in tqdm(self.ds, "Evaluating " + self.test_names, total=self.steps):
            emb = self.strategy.run(lambda x: self.basic_model(x, training=False), args=(img_batch,))
            emb = tf.concat(emb.values, axis=0)
            if self.flip:
                emb_f = self.strategy.run(lambda x: self.basic_model(tf.image.flip_left_right(x), training=False), args=(img_batch,))
                emb_f = tf.concat(emb_f.values, axis=0)
                emb = emb + emb_f
            embs.append(emb.numpy())
        return np.concatenate(embs, axis=0)

    def __eval_func__(self, cur_step=0, logs=None, eval_freq=1):
        if cur_step % eval_freq != 0:
            return
        if eval_freq > 1:
            if cur_step == 0:
                return
            cur_epoch = self.model.history.epoch[-1] if self.model and len(self.model.history.__dict__.get("epoch", [])) else 0
            cur_step = f"{cur_epoch + 1}_batch_{cur_step}"
        else:
            cur_step = str(cur_step + 1)
        
        dists = []
        print("")  # New line for readability
        if self.is_distribute:
            embs = self.__do_predict_distribute__()
        else:
            embs = self.__do_predict__()

        if not np.all(np.isfinite(embs)):
            print("NAN in embs, not a good one")
            return
        self.embs = embs
        embs = normalize(embs)
        embs_a = embs[::2]
        embs_b = embs[1::2]
        dists = (embs_a * embs_b).sum(1)

        tt = np.sort(dists[self.test_issame[:dists.shape[0]]])
        ff = np.sort(dists[np.logical_not(self.test_issame[:dists.shape[0]])])
        self.tt, self.ff = tt, ff

        t_steps = int(0.1 * ff.shape[0])
        acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
        acc_max_indx = np.argmax(acc_count)
        acc_max = acc_count[acc_max_indx] / dists.shape[0]
        self.acc_thresh = ff[acc_max_indx - t_steps]
        self.cur_acc = acc_max

        if self.PCA_acc:
            _, _, accuracy, val, val_std, far = evaluate(embs, self.test_issame, nrof_folds=10)
            acc2, std2 = np.mean(accuracy), np.std(accuracy)
            print(
                f"\n>>>> {self.test_names} evaluation max accuracy: {acc_max}, thresh: {self.acc_thresh}, previous max accuracy: {self.max_accuracy}, PCA accuracy = {acc2} ± {std2}"
            )
        else:
            print(
                f"\n>>>> {self.test_names} evaluation max accuracy: {acc_max}, thresh: {self.acc_thresh}, previous max accuracy: {self.max_accuracy}"
            )

        if acc_max >= self.max_accuracy:
            print(f">>>> Improved = {acc_max - self.max_accuracy}")
            self.max_accuracy = acc_max
            if self.save_model:
                save_name_base = f"{self.save_model}_basic_{self.test_names}_epoch_"
                save_path_base = os.path.join("checkpoints", save_name_base)
                # Uncomment to clean old models: for ii in glob2.glob(save_path_base + "*.h5"): os.remove(ii)
                save_path = f"{save_path_base}{cur_step}_{self.max_accuracy:.4f}.h5"
                print(f"Saving model to: {save_path}")
                self.basic_model.save(save_path, include_optimizer=False)

def half_split_weighted_cosine_similarity_11(aa, bb):
    half = aa.shape[-1] // 2
    bb = bb[:aa.shape[0]]

    top_weights = tf.norm(aa[:, :half], axis=1) * tf.norm(bb[:, :half], axis=1)
    bottom_weights = tf.norm(aa[:, half:], axis=1) * tf.norm(bb[:, half:], axis=1)

    top_sim = tf.reduce_sum(aa[:, :half] * bb[:, :half], axis=-1)
    bottom_sim = tf.reduce_sum(aa[:, half:] * bb[:, half:], axis=-1)
    return (top_sim + bottom_sim) / (top_weights + bottom_weights + 1e-6)  # Add small epsilon to avoid division by zero

def half_split_weighted_cosine_similarity(aa, bb):
    half = aa.shape[-1] // 2
    bb = tf.transpose(bb)

    top_weights = tf.norm(aa[:, :half], axis=-1, keepdims=True) * tf.norm(bb[:half], axis=0, keepdims=True)
    bottom_weights = tf.norm(aa[:, half:], axis=-1, keepdims=True) * tf.norm(bb[half:], axis=0, keepdims=True)

    top_sim = aa[:, :half] @ bb[:half]
    bottom_sim = aa[:, half:] @ bb[half:]
    return (top_sim + bottom_sim) / (top_weights + bottom_weights + 1e-6)  # Add small epsilon to avoid division by zero

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    acc = float(tp + tn) / dist.size if dist.size > 0 else 0.0
    return tpr, fpr, acc

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind="slinear", fill_value="extrapolate")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same) if n_same > 0 else 0.0
    far = float(false_accept) / float(n_diff) if n_diff > 0 else 0.0
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

if __name__ == "__main__":
    import sys
    import argparse
    import tensorflow_addons as tfa

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--basic_model", type=str, required=True, help="Model file, keras h5")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-t", "--test_bin_files", nargs="*", type=str, help="Test bin files")
    parser.add_argument("-F", "--no_flip", action="store_true", help="Disable flip")
    args = parser.parse_known_args(sys.argv[1:])[0]

    # Suppress TensorFlow logging if needed
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.get_logger().setLevel('ERROR')

    basic_model = tf.keras.models.load_model(args.basic_model, compile=False)
    flip = not args.no_flip
    for test_bin_file in args.test_bin_files:
        if not os.path.exists(test_bin_file):
            print(f"Error: File not found: {test_bin_file}")
            continue
        aa = eval_callback(basic_model, test_bin_file, batch_size=args.batch_size, flip=flip)
        aa.on_epoch_end()
elif __name__ == "__test__":
    from data_distiller import teacher_model_interf_wrapper

    mm = teacher_model_interf_wrapper("C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\GhostFaceNets\\checkpoints\\train_epoch01.h5")
    evals.eval_callback(lambda imm: mm(imm * 128 + 127.5), "C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\datasets\\faces_umd\\faces_umd/agedb_30.bin").on_epoch_end()