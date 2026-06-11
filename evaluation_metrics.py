import numpy as np
from scipy.stats import spearmanr, kendalltau, rankdata


# =========================
# PGL-SUM STYLE F1
# =========================
def evaluate_summary(predicted_summary, user_summary, eval_method="max"):

    if predicted_summary is None or user_summary is None:
        return 0.0

    max_len = max(len(predicted_summary), user_summary.shape[1])

    S = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = np.asarray(predicted_summary).astype(int)

    f_scores = []

    for user in range(user_summary.shape[0]):

        G = np.zeros(max_len, dtype=int)
        G[:user_summary.shape[1]] = user_summary[user]

        overlapped = S & G

        precision = np.sum(overlapped) / np.sum(S) if np.sum(S) > 0 else 0
        recall = np.sum(overlapped) / np.sum(G) if np.sum(G) > 0 else 0

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(
                2 * precision * recall * 100 / (precision + recall)
            )

    return np.max(f_scores) if eval_method == "max" else np.mean(f_scores)


# =========================
# CORRELATION + F1
# =========================
def get_corr_coeff(
    pred_imp_scores,
    videos,
    dataset,
    user_scores=None,
    generated_summary=None,
    positions=None,
    change_points=None,
    n_frames=None,
    user_summary=None
):

    rho_coeff, tau_coeff, f1_coeff = [], [], []

    # =========================
    # SUMME
    # =========================
    if dataset == 'SumMe':

        for pred_imp_score, video in zip(pred_imp_scores, videos):

            pred_imp_score = np.asarray(pred_imp_score).flatten()

            # ---- SAFE GT ----
            if user_summary is not None:
                true = np.mean(user_summary, axis=0)
            else:
                true = None

            if true is None or len(true) == 0:
                rho_coeff.append(0.0)
                tau_coeff.append(0.0)
            else:
                min_len = min(len(pred_imp_score), len(true))

                pred = pred_imp_score[:min_len]
                gt = true[:min_len]

                rho, _ = spearmanr(pred, gt)
                tau, _ = kendalltau(rankdata(pred), rankdata(gt))

                rho_coeff.append(0.0 if np.isnan(rho) else rho)
                tau_coeff.append(0.0 if np.isnan(tau) else tau)

            # F1
            f1_coeff.append(
                evaluate_summary(generated_summary, user_summary, "max")
            )

    # =========================
    # TVSUM
    # =========================
    elif dataset == 'TVSum':

        if user_scores is None:
            user_scores = []

        for pred_imp_score, video in zip(pred_imp_scores, videos):

            pred_imp_score = np.asarray(pred_imp_score).flatten()

            user_id = int(video.split("_")[-1]) - 1

            if user_id < 0 or user_id >= len(user_scores):
                rho_coeff.append(0.0)
                tau_coeff.append(0.0)
                f1_coeff.append(0.0)
                continue

            curr_user_score = user_scores[user_id]

            tmp_rho, tmp_tau = [], []

            for annotation in curr_user_score:

                true_user_score = np.asarray(annotation).flatten()

                min_len = min(len(pred_imp_score), len(true_user_score))

                pred = pred_imp_score[:min_len]
                gt = true_user_score[:min_len]

                rho, _ = spearmanr(pred, gt)
                tau, _ = kendalltau(rankdata(pred), rankdata(gt))

                tmp_rho.append(0.0 if np.isnan(rho) else rho)
                tmp_tau.append(0.0 if np.isnan(tau) else tau)

            rho_coeff.append(np.mean(tmp_rho))
            tau_coeff.append(np.mean(tmp_tau))

            # F1
            f1_coeff.append(
                evaluate_summary(generated_summary, user_summary, "avg")
                if user_summary is not None else 0.0
            )

    return rho_coeff, tau_coeff, f1_coeff